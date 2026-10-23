#!/usr/bin/env python3
"""Phase 296: does a THIRD, genuinely different protein language model add to the confirmed frozen ESM2-650M view?
ProtT5-XL-U50 (Rostlab/prot_t5_xl_half_uniref50-enc: T5 encoder, BFD + UniRef50, 1024-d) as a second zero-initialised frozen branch.
Instrument: the Phase 295 4-fold CV (760 held-out chains, fixed 12 epochs, test untouched), reusing Phase 295's base and frozen650
predictions as comparators; a base seed-0 re-run must reproduce Phase 295 bit-for-bit or the comparison is void. (It did not: GPU
scatter-adds are not bitwise deterministic, held-out PR moved 0.0001. Variant 296b therefore re-runs all four arms in one kernel.)
PRE-REGISTERED: PRIMARY = pooled held-out PR-AUC, (a) prott5 vs base, (b) frozen650_prott5 vs frozen650, paired by seed, t AND Wilcoxon
< .05 after Holm over the two. Secondary: ROC, per-protein AP, P@20.
usage: build_nb296.py <output_root>"""
import json, subprocess, sys, tempfile
from pathlib import Path

ROOT_OUT = Path(sys.argv[1]).resolve(); SLUG = "296"
HERE = Path(__file__).resolve().parent
tmp = Path(tempfile.mkdtemp())
subprocess.run([sys.executable, str(HERE.parent / "phase295" / "build_nb295.py"), str(tmp)], check=True, capture_output=True)
nb = json.load(open(tmp / "295" / "phase295_cv_instrument.ipynb"))
cells = nb["cells"]
src = lambda i: cells[i]["source"] if isinstance(cells[i]["source"], str) else "".join(cells[i]["source"])
def setsrc(i, s): cells[i]["source"] = s
def rep(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)
find = lambda key: next(i for i in range(len(cells)) if key in src(i))

setsrc(0, "# Phase 296 - a third language model (ProtT5-XL) as a second frozen branch, on the Phase 295 CV instrument\n")
i = find("pip install"); setsrc(i, rep(src(i), "biopython==1.84", "biopython==1.84 sentencepiece"))
i = find("frozen 650M embeddings for")
setsrc(i, src(i) + r'''

from transformers import T5EncoderModel, T5Tokenizer
T5_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"
t5tok = T5Tokenizer.from_pretrained(T5_NAME, do_lower_case=False)
t5 = T5EncoderModel.from_pretrained(T5_NAME, torch_dtype=torch.float16).to(DEVICE).eval()
@torch.no_grad()
def t5_embed(seq):
    n = -(-len(seq) // CHUNK); size = -(-len(seq) // n); outs = []
    for i in range(0, len(seq), size):
        part = seq[i:i + size]
        ids = t5tok(" ".join(part), return_tensors="pt").to(DEVICE)
        assert ids["input_ids"].shape[1] == len(part) + 1          # one token per residue + </s>
        outs.append(t5(**ids).last_hidden_state[0, :len(part)].float())
    return torch.cat(outs)
t_ = time.time()
for d in dataset:
    d.plm2_emb = t5_embed(seq_of(str(d.pdb_id), str(d.chain_id))).half().cpu(); assert d.plm2_emb.shape == (d.num_nodes, 1024)
del t5; torch.cuda.empty_cache(); print(f"ProtT5 embeddings for {len(dataset)} chains [{(time.time()-t_)/60:.1f}m]")
''')
i = find("class EGCL"); m = src(i)
m = rep(m, "use_loc_head=True, plm_dim=None):", "use_loc_head=True, plm_dim=None, plm2_dim=None):")
m = rep(m, "            s.plm_proj=nn.Linear(plm_dim,h); nn.init.zeros_(s.plm_proj.weight); nn.init.zeros_(s.plm_proj.bias)\n",
        "            s.plm_proj=nn.Linear(plm_dim,h); nn.init.zeros_(s.plm_proj.weight); nn.init.zeros_(s.plm_proj.bias)\n"
        "        s.plm2_dim=plm2_dim\n"
        "        if plm2_dim:\n"
        "            s.plm2_proj=nn.Linear(plm2_dim,h); nn.init.zeros_(s.plm2_proj.weight); nn.init.zeros_(s.plm2_proj.bias)\n")
m = rep(m, "        if s.plm_dim: h=h+s.plm_proj(d.plm_emb.float())\n",
        "        if s.plm_dim: h=h+s.plm_proj(d.plm_emb.float())\n        if s.plm2_dim: h=h+s.plm2_proj(d.plm2_emb.float())\n")
setsrc(i, m)
i = find("def fit_fixed"); t = src(i)
t = rep(t, "def fit_fixed(train_set, seed, pretrained_path, plm_dim, mode, kappa, epochs=12, lam=0.1, batch=16):",
        "def fit_fixed(train_set, seed, pretrained_path, plm_dim, mode, kappa, epochs=12, lam=0.1, batch=16, plm2_dim=None):")
t = rep(t, "LocalizingEGNN(N_PRE, SIDE_DIM, use_loc_head=True, plm_dim=plm_dim)", "LocalizingEGNN(N_PRE, SIDE_DIM, use_loc_head=True, plm_dim=plm_dim, plm2_dim=plm2_dim)")
t = rep(t, 'if not k.startswith("plm_proj")', 'if not k.startswith(("plm_proj", "plm2_proj"))')
setsrc(i, t)
i = find("ARMS = [")
setsrc(i, r'''
INKERNEL = INKERNEL_FLAG
KEEP_SIDE = list(range(TOPO_DIM)); m_ = torch.zeros(SIDE_DIM); m_[KEEP_SIDE] = 1.0
for d in dataset: d.x[:, N_PRE:] = d.x[:, N_PRE:] * m_
ARMS = ([("base_check", None, None, [0]), ("prott5", None, 1024, list(range(8))), ("frozen650_prott5", 1280, 1024, list(range(8)))] if not INKERNEL else
        [(a, p1, p2, list(range(8))) for a, p1, p2 in (("base", None, None), ("frozen650", 1280, None), ("prott5", None, 1024), ("frozen650_prott5", 1280, 1024))])
T0 = time.time(); BUDGET = 10.5 * 3600; ALL = {}
for name, plm_dim, plm2_dim, seeds in ARMS:
    for seed in seeds:
        if time.time() - T0 > BUDGET: print(f"budget reached before {name} seed {seed}"); break
        ck = locate(f"pretrain266_full_seed{seed}.pt"); preds = {}
        for f in range(4):
            model = fit_fixed([d for d in CV if d.fold != f], seed, ck, plm_dim, "bce", 1.0, plm2_dim=plm2_dim)
            preds.update(predict(model, [d for d in CV if d.fold == f]))
        ALL.setdefault(name, {})[seed] = preds
        torch.save(ALL, WORK / "phase296_cv_preds.pt")
        pp = np.concatenate([v["p"] for v in preds.values()]); yy = np.concatenate([v["y"] for v in preds.values()])
        print(f"  seed {seed} [{name}] held-out PR {average_precision_score(yy, pp):.4f}  [{(time.time()-T0)/60:.1f}m]", flush=True)
print("done")
''')
import copy
for slug, flag, kid, title in (("296", "False", "phase296-prott5-cv", "Phase296 Prott5 Cv"), ("296b", "True", "phase296b-prott5-cv-inkernel", "Phase296b Prott5 Cv Inkernel")):
    nbv = copy.deepcopy(nb); j = next(i for i in range(len(nbv["cells"])) if "INKERNEL_FLAG" in (nbv["cells"][i]["source"] if isinstance(nbv["cells"][i]["source"], str) else "".join(nbv["cells"][i]["source"])))
    sj = nbv["cells"][j]["source"]; nbv["cells"][j]["source"] = (sj if isinstance(sj, str) else "".join(sj)).replace("INKERNEL_FLAG", flag)
    d = ROOT_OUT / slug; d.mkdir(parents=True, exist_ok=True)
    o = d / f"phase{slug}_prott5_cv.ipynb"; o.write_text(json.dumps(nbv, indent=1) + "\n")
    meta = json.load(open(tmp / "295" / "kernel-metadata.json"))
    meta.update(id=f"blackpearls1/{kid}", title=title, code_file=o.name)
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", o)
