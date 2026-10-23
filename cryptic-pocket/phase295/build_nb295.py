#!/usr/bin/env python3
"""Phase 295: a stronger instrument. 4-fold cross-validation on CryptoBench's own homology-clean training folds (train-0..3), so all
760 train+val chains are held-out evaluation and the test set is never touched. Built because two leads (frozen650, Phase 293; pu20,
Phase 294) moved the 182-chain TEST set significantly but not the 60-chain validation set they were pre-registered on.

PRE-REGISTERED before any result: for each seed s (0-7) and fold f, fine-tune from pretrained seed s on the other three folds for a
FIXED 12 epochs (no early stopping: the median stopping epoch seen in Phases 291-294), predict fold f. PRIMARY = PR-AUC pooled over
all 760 held-out chains (official union labels), each arm vs base paired by seed, t-test AND Wilcoxon both < .05 after Holm over the 3
arms. Secondary: held-out ROC, per-protein AP, precision@20. Arms:
  base            Phase 291 recipe
  pu20            non-negative PU loss, kappa = 2 (Phase 294)
  frozen650       + zero-initialised frozen ESM2-650M branch (Phase 293)
  frozen650_pu20  both
usage: build_nb295.py <output_root>"""
import json, subprocess, sys, tempfile
from pathlib import Path

ROOT_OUT = Path(sys.argv[1]).resolve(); SLUG = "295"
HERE = Path(__file__).resolve().parent
tmp = Path(tempfile.mkdtemp())
subprocess.run([sys.executable, str(HERE.parent / "phase291" / "build_nb291.py"), str(tmp)], check=True, capture_output=True)
nb291 = json.load(open(tmp / "291" / "phase291_cb3b_official_labels.ipynb"))
src = lambda i: nb291["cells"][i]["source"] if isinstance(nb291["cells"][i]["source"], str) else "".join(nb291["cells"][i]["source"])
assert "resnums_of" in src(6) and "class EGCL" in src(8) and "def run_epoch" in src(10)

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

def rep(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)

md('''
# Phase 295 - 4-fold cross-validation instrument on CryptoBench's training folds (test set untouched)

Arms `base`, `pu20`, `frozen650`, `frozen650_pu20`; 8 seeds x 4 folds each; fixed 12 epochs; primary = pooled held-out PR-AUC over 760 chains.
''')
for i in range(1, 7):
    (md if nb291["cells"][i]["cell_type"] == "markdown" else code)(src(i))
md("## Cell 4 - Fold assignment (CryptoBench folds.json) for the 760 train+val chains")
code(r'''
cb_folds = fetch_json("https://osf.io/download/5s93p/")
fold_of = {pid: int(k.split("-")[1]) for k, v in cb_folds.items() if k.startswith("train-") for pid in v}
CV = [d for d in dataset if d.split in ("train", "val")]
for d in CV: d.fold = fold_of[str(d.pdb_id)]
for f in range(4):
    ch = [d for d in CV if d.fold == f]
    print(f"  fold {f}: {len(ch)} chains, {sum(d.num_nodes for d in ch):,} residues, {sum(int(d.y.sum()) for d in ch):,} positives")
assert len(CV) == 760
''')
md("## Cell 5 - Sequences + frozen ESM2-650M embeddings (Phase 293 branch, frozen)")
code(r'''
AA1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K",
       "MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M"}
def seq_of(pid, cid):
    path = paths[pid]
    parser = MMCIFParser(QUIET=True) if ".cif" in path.name else PDBParser(QUIET=True, PERMISSIVE=True)
    with gzip.open(path, "rt") as fh: model = next(parser.get_structure(pid, fh).get_models())
    return "".join(AA1[r.get_resname().strip().upper()] for r in model[cid] if r.get_resname().strip().upper() in AA1 and "CA" in r)
from transformers import AutoTokenizer, EsmModel
PLM_NAME = "facebook/esm2_t33_650M_UR50D"; CHUNK = 1000
tok = AutoTokenizer.from_pretrained(PLM_NAME)
plm0 = EsmModel.from_pretrained(PLM_NAME, add_pooling_layer=False).to(DEVICE).eval()
@torch.no_grad()
def plm_embed(seq):
    n = -(-len(seq) // CHUNK); size = -(-len(seq) // n); outs = []
    for i in range(0, len(seq), size):
        ids = tok(seq[i:i + size], return_tensors="pt").to(DEVICE)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            outs.append(plm0(**ids).last_hidden_state[0, 1:-1].float())
    return torch.cat(outs)
t_ = time.time()
for d in dataset:
    s = seq_of(str(d.pdb_id), str(d.chain_id)); assert len(s) == d.num_nodes
    d.plm_emb = plm_embed(s).half().cpu()
del plm0; torch.cuda.empty_cache(); print(f"frozen 650M embeddings for {len(dataset)} chains [{(time.time()-t_)/60:.1f}m]")
''')
md("## Cell 6 - Model (Phase 129) + optional zero-initialised language-model branch (Phase 293)")
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
md("## Cell 7 - Training: fixed epochs, no early stopping; BCE or non-negative PU loss")
code(r'''
def run_epoch(model, loader, opt, lam, mode, kappa):
    model.train(True)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE)); huber = nn.SmoothL1Loss(beta=4.0)
    for b in loader:
        b = b.to(DEVICE)
        logits, pred_c = model(b)
        if mode == "pu":
            P_ = b.y > 0; n_ = logits.numel(); ln_ = nn.functional.softplus(logits)
            loss = POS_WEIGHT * nn.functional.softplus(-logits)[P_].sum() / n_ + torch.clamp((ln_.sum() - kappa * ln_[P_].sum()) / n_, min=0.0)
        else:
            loss = crit(logits, b.y)
        if lam > 0 and pred_c is not None: loss = loss + lam * huber(pred_c, b.true_center.view(-1, 3))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

def fit_fixed(train_set, seed, pretrained_path, plm_dim, mode, kappa, epochs=12, lam=0.1, batch=16):
    torch.manual_seed(seed); np.random.seed(seed)
    model = LocalizingEGNN(N_PRE, SIDE_DIM, use_loc_head=True, plm_dim=plm_dim).to(DEVICE)
    missing, unexpected = model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE), strict=False)
    assert not [k for k in missing if not k.startswith("plm_proj")] and not unexpected, (missing, unexpected)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tl = DataLoader(train_set, batch_size=batch, shuffle=True)
    for ep in range(epochs): run_epoch(model, tl, opt, lam, mode, kappa); sch.step()
    return model.eval()

@torch.no_grad()
def predict(model, chains):
    out = {}
    for d in chains:
        b = d.clone().to(DEVICE); logits, pc = model(b)
        out[(str(d.pdb_id).lower(), str(d.chain_id))] = dict(p=torch.sigmoid(logits).cpu().numpy().astype(np.float32), y=d.y.numpy().astype(np.int8),
                                                             fold=int(d.fold), c=pc.cpu().numpy()[0], tc=d.true_center.numpy())
    return out
''')
md("## Cell 8 - Arms x 8 seeds x 4 folds; incremental save; 10.5 h budget guard")
code(r'''
KEEP_SIDE = list(range(TOPO_DIM)); m_ = torch.zeros(SIDE_DIM); m_[KEEP_SIDE] = 1.0        # arm "full", as in 266c/291
for d in dataset: d.x[:, N_PRE:] = d.x[:, N_PRE:] * m_
ARMS = [("base", None, "bce", 1.0), ("pu20", None, "pu", 2.0), ("frozen650", 1280, "bce", 1.0), ("frozen650_pu20", 1280, "pu", 2.0)]
T0 = time.time(); BUDGET = 10.5 * 3600; ALL = {}
for seed in range(8):
    if time.time() - T0 > BUDGET: print(f"budget reached before seed {seed}"); break
    ck = locate(f"pretrain266_full_seed{seed}.pt")
    for name, plm_dim, mode, kappa in ARMS:
        preds = {}
        for f in range(4):
            model = fit_fixed([d for d in CV if d.fold != f], seed, ck, plm_dim, mode, kappa)
            preds.update(predict(model, [d for d in CV if d.fold == f]))
        ALL.setdefault(name, {})[seed] = preds
        torch.save(ALL, WORK / "phase295_cv_preds.pt")
        pp = np.concatenate([v["p"] for v in preds.values()]); yy = np.concatenate([v["y"] for v in preds.values()])
        print(f"  seed {seed} [{name}] held-out PR {average_precision_score(yy, pp):.4f} over {len(preds)} chains  [{(time.time()-T0)/60:.1f}m]", flush=True)
print("done")
''')
nb = {"cells": C, "metadata": nb291["metadata"], "nbformat": 4, "nbformat_minor": 5}
d = ROOT_OUT / SLUG; d.mkdir(parents=True, exist_ok=True)
o = d / "phase295_cv_instrument.ipynb"; o.write_text(json.dumps(nb, indent=1) + "\n")
meta = {"id": "blackpearls1/phase295-cv-instrument", "title": "Phase295 Cv Instrument", "code_file": o.name,
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4",
        "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["blackpearls1/tda-egnn-phase48-cryptobench-esm3b", "blackpearls1/phase227-geometry-ingestion", "blackpearls1/phase266b-pretrain-3b-clean"]}
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote", o)
