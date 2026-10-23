#!/usr/bin/env python3
"""Phase 293: fine-tune a protein language model end to end (LoRA on ESM2-650M) inside the Phase 291 recipe.

The existing model reads FROZEN, precomputed ESM2-3B embeddings. Here an ESM2-650M branch runs inside the training loop: its
per-residue output goes through a ZERO-INITIALISED projection added to the EGNN's first hidden state, so at step 0 the model is
exactly the pretrained Phase 266b/291 model and the branch only adds what it learns. Arms (8 seeds each, seed s <- pretrained s):
  frozen650: the branch with ESM2-650M frozen (embeddings computed once) -> effect of a second language-model view
  lora650:   the branch with LoRA (r=8, alpha=16, dropout .05, query/key/value of every layer) trained jointly -> effect of fine-tuning
Base = Phase 292's in-kernel base (bit-identical to Phase 291). Official union labels, same checkpoints, seeds, split, loss,
early stopping (val PR, patience 8, <=40 epochs), batch 16. LoRA params lr 2e-4 (EGNN + projection 1e-3, as before); the PLM runs
under fp16 autocast with a GradScaler. Long chains are embedded in equal chunks of <=1000 residues.
Kernels: a = frozen650 (8 seeds) + lora650 seeds 0-3; b = lora650 seeds 4-7. A 10.5 h budget guard stops new seeds.
usage: build_nb293.py <output_root>"""
import json, subprocess, sys, tempfile
from pathlib import Path

ROOT_OUT = Path(sys.argv[1]).resolve()
HERE = Path(__file__).resolve().parent
tmp = Path(tempfile.mkdtemp())
subprocess.run([sys.executable, str(HERE.parent / "phase291" / "build_nb291.py"), str(tmp)], check=True, capture_output=True)
nb291 = json.load(open(tmp / "291" / "phase291_cb3b_official_labels.ipynb"))
src = lambda i: nb291["cells"][i]["source"] if isinstance(nb291["cells"][i]["source"], str) else "".join(nb291["cells"][i]["source"])
assert "pip install" in src(2) and "resnums_of" in src(6) and "class EGCL" in src(8)


def rep(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)


def build(slug, do_frozen, lora_seeds):
    C = []
    md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
    code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})
    md(f'''
# Phase 293{slug} - LoRA fine-tuning of ESM2-650M inside the Phase 291 recipe

This kernel: frozen650 arm {"(8 seeds)" if do_frozen else "(not run here)"}; lora650 seeds {lora_seeds}.
''')
    md("## Cell 1 - Install")
    code(rep(src(2), "biopython==1.84", "biopython==1.84 peft==0.13.2"))
    for i in range(3, 7):                                  # setup + official relabel (Phase 291 verbatim)
        (md if nb291["cells"][i]["cell_type"] == "markdown" else code)(src(i))
    md("## Cell 4 - Sequences in node order (Phase 48 residue rule) + ESM2-650M")
    code(r'''
AA1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K",
       "MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M"}
def seq_of(pid, cid):
    path = paths[pid]
    parser = MMCIFParser(QUIET=True) if ".cif" in path.name else PDBParser(QUIET=True, PERMISSIVE=True)
    with gzip.open(path, "rt") as fh: model = next(parser.get_structure(pid, fh).get_models())
    return "".join(AA1[r.get_resname().strip().upper()] for r in model[cid] if r.get_resname().strip().upper() in AA1 and "CA" in r)
for d in dataset:
    d.seq = seq_of(str(d.pdb_id), str(d.chain_id)); assert len(d.seq) == d.num_nodes
print("sequences attached; longest", max(len(d.seq) for d in dataset))

from transformers import AutoTokenizer, EsmModel
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
PLM_NAME = "facebook/esm2_t33_650M_UR50D"; PLM_DIM = 1280; CHUNK = 1000
tok = AutoTokenizer.from_pretrained(PLM_NAME)
_t = tok("MKTAYIAK", return_tensors="pt")["input_ids"]; assert _t.shape[1] == 8 + 2, _t     # one token per residue + BOS/EOS

def load_plm(lora):
    m = EsmModel.from_pretrained(PLM_NAME, add_pooling_layer=False).to(DEVICE)
    if lora:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})      # activations of 16 seqs x 33 layers
        m = get_peft_model(m, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["query", "key", "value"], bias="none"))
        n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad); print(f"LoRA trainable params {n_tr:,}")
    else:
        for p in m.parameters(): p.requires_grad_(False)
    return m

def plm_embed(plm, seqs, grad):
    outs = []
    for s in seqs:
        n = -(-len(s) // CHUNK); size = -(-len(s) // n)
        for i in range(0, len(s), size):
            ids = tok(s[i:i + size], return_tensors="pt").to(DEVICE)
            with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                outs.append(plm(**ids).last_hidden_state[0, 1:-1].float())
    return torch.cat(outs)
''')
    md("## Cell 5 - Model (Phase 129) + zero-initialised language-model branch")
    m = src(8)
    m = rep(m, "def __init__(s, n_pre, td, h=128, L=4, dr=0.1, use_loc_head=True):", "def __init__(s, n_pre, td, h=128, L=4, dr=0.1, use_loc_head=True, plm_dim=None):")
    m = rep(m, "        s.head=nn.Sequential(nn.Linear(h,h//2),nn.SiLU(),nn.Dropout(dr),nn.Linear(h//2,1))\n",
            "        s.head=nn.Sequential(nn.Linear(h,h//2),nn.SiLU(),nn.Dropout(dr),nn.Linear(h//2,1))\n"
            "        s.plm_dim=plm_dim\n"
            "        if plm_dim:\n"
            "            s.plm_proj=nn.Linear(plm_dim,h); nn.init.zeros_(s.plm_proj.weight); nn.init.zeros_(s.plm_proj.bias)\n")
    m = rep(m, "        h=s.embed(d.x); pos=d.pos; t=d.x[:,s.n_pre:]\n",
            "        h=s.embed(d.x); pos=d.pos; t=d.x[:,s.n_pre:]\n"
            "        if s.plm_dim: h=h+s.plm_proj(d.plm_emb.float())\n")
    code(m)
    md("## Cell 6 - Training (Phase 129 semantics; the PLM is embedded per batch when LoRA is on)")
    code(r'''
def run_epoch(model, loader, opt=None, lam=0.0, plm=None, scaler=None):
    tr = opt is not None; model.train(tr)
    if plm is not None: plm.train(tr)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))
    huber = nn.SmoothL1Loss(beta=4.0)
    ps, ys = [], []
    for b in loader:
        b = b.to(DEVICE)
        with torch.set_grad_enabled(tr):
            if plm is not None: b.plm_emb = plm_embed(plm, b.seq, tr)
            logits, pred_c = model(b)
            loss = crit(logits, b.y)
            if lam > 0 and pred_c is not None: loss = loss + lam * huber(pred_c, b.true_center.view(-1, 3))
        if tr:
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0); scaler.step(opt); scaler.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        ps.append(torch.sigmoid(logits).detach().cpu().numpy()); ys.append(b.y.cpu().numpy())
    p, y = np.concatenate(ps), np.concatenate(ys)
    return {"pr": average_precision_score(y, p), "roc": roc_auc_score(y, p)}

def fit(seed, pretrained_path, lora, lam=0.1, epochs=40, patience=8, batch=16):
    torch.manual_seed(seed); np.random.seed(seed)
    model = LocalizingEGNN(N_PRE, SIDE_DIM, use_loc_head=True, plm_dim=PLM_DIM).to(DEVICE)
    missing, unexpected = model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE), strict=False)
    assert not [k for k in missing if not k.startswith("plm_proj")] and not unexpected, (missing, unexpected)
    plm = load_plm(True) if lora else None
    groups = [{"params": list(model.parameters()), "lr": 1e-3}]
    if lora: groups.append({"params": [p for p in plm.parameters() if p.requires_grad], "lr": 2e-4, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler() if lora else None
    tl = DataLoader(splits["train"], batch_size=batch, shuffle=True); vl = DataLoader(splits["val"], batch_size=batch)
    best, bad, state_best, lora_best, ep_used = -1, 0, None, None, 0
    for ep in range(epochs):
        run_epoch(model, tl, opt, lam, plm, scaler); v = run_epoch(model, vl, lam=lam, plm=plm); sch.step(); ep_used = ep + 1
        if v["pr"] > best:
            best, bad = v["pr"], 0
            state_best = {k: t.detach().clone() for k, t in model.state_dict().items()}
            if lora: lora_best = {k: t.detach().clone() for k, t in get_peft_model_state_dict(plm).items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(state_best); model.eval()
    if lora: set_peft_model_state_dict(plm, lora_best); plm.eval()
    return model, plm, best, ep_used

@torch.no_grad()
def predict(model, plm, split):
    out = {}
    for d in splits[split]:
        b = d.clone().to(DEVICE)
        if plm is not None: b.plm_emb = plm_embed(plm, [d.seq], False)
        logits, pc = model(b)
        out[(str(d.pdb_id).lower(), str(d.chain_id))] = dict(p=torch.sigmoid(logits).cpu().numpy().astype(np.float32), c=pc.cpu().numpy()[0],
            y=d.y.numpy().astype(np.int8), y_main=d.y_main.numpy().astype(np.int8), tc=d.true_center.numpy())
    return out
''')
    md("## Cell 7 - Run the arms; incremental save after every seed; 10.5 h budget guard")
    code(f'''
DO_FROZEN = {do_frozen}; LORA_SEEDS = {lora_seeds}; SLUG = "{slug}"
''' + r'''
KEEP_SIDE = list(range(TOPO_DIM)); m_ = torch.zeros(SIDE_DIM); m_[KEEP_SIDE] = 1.0      # arm "full", as in 266c/291
for d in dataset: d.x[:, N_PRE:] = d.x[:, N_PRE:] * m_
T0 = time.time(); BUDGET = 10.5 * 3600

if DO_FROZEN:
    plm0 = load_plm(False); plm0.eval()
    with torch.no_grad():
        for d in dataset: d.plm_emb = plm_embed(plm0, [d.seq], False).half().cpu()
    del plm0; torch.cuda.empty_cache(); print(f"frozen 650M embeddings cached [{(time.time()-T0)/60:.1f}m]", flush=True)
    PREDS = {"arm": "frozen650", "seeds": {}}
    for seed in range(8):
        model, _, val_pr, ep = fit(seed, locate(f"pretrain266_full_seed{seed}.pt"), lora=False)
        PREDS["seeds"][seed] = {"val": predict(model, None, "val"), "test": predict(model, None, "test"), "val_pr": float(val_pr), "epochs": ep}
        torch.save(PREDS, WORK / "phase293_frozen650_preds.pt")
        print(f"  [frozen650] seed {seed}: val PR {val_pr:.4f} ({ep} ep)  [{(time.time()-T0)/60:.1f}m]", flush=True)
    for d in dataset: del d.plm_emb

PREDS = {"arm": "lora650", "seeds": {}}
for seed in LORA_SEEDS:
    if time.time() - T0 > BUDGET: print(f"budget reached; seed {seed} not started"); break
    t1 = time.time()
    model, plm, val_pr, ep = fit(seed, locate(f"pretrain266_full_seed{seed}.pt"), lora=True)
    PREDS["seeds"][seed] = {"val": predict(model, plm, "val"), "test": predict(model, plm, "test"), "val_pr": float(val_pr), "epochs": ep}
    torch.save(PREDS, WORK / f"phase293{SLUG}_lora650_preds.pt")
    print(f"  [lora650] seed {seed}: val PR {val_pr:.4f} ({ep} ep, {(time.time()-t1)/60:.1f}m)  [{(time.time()-T0)/60:.1f}m]", flush=True)
    del model, plm; torch.cuda.empty_cache()
print("done")
''')
    nb = {"cells": C, "metadata": nb291["metadata"], "nbformat": 4, "nbformat_minor": 5}
    d = ROOT_OUT / f"293{slug}"; d.mkdir(parents=True, exist_ok=True)
    o = d / f"phase293{slug}_lora_esm650.ipynb"; o.write_text(json.dumps(nb, indent=1) + "\n")
    meta = {"id": f"blackpearls1/phase293{slug}-lora-esm650", "title": f"Phase293{slug} Lora Esm650", "code_file": o.name,
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4",
            "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
            "kernel_sources": ["blackpearls1/tda-egnn-phase48-cryptobench-esm3b", "blackpearls1/phase227-geometry-ingestion", "blackpearls1/phase266b-pretrain-3b-clean"]}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", o)


build("a", True, [0, 1, 2, 3])
build("b", False, [4, 5, 6, 7])
