#!/usr/bin/env python3
"""Phase 309: does extra cryptic supervision from CryptoBank help? Phase 308d built 632 extra graphs (one per 95% cluster, no
CryptoBank cluster has ANY MMseqs2 e < 1e-3 hit to any CryptoBench chain) in the Phase 48 node format. They are never held out and
never used for early stopping: every held-out fold and every inner early-stopping set is pure CryptoBench, as in Phase 306.
Arms (stack3 = ESM2-3B input + 650M + ProtT5 + SaProt branches, Phase 306 deployment-matched instrument, 8 seeds x 4 folds):
  stack3          comparator
  stack3_mix      the 632 extra chains appended to each fold's training chains (same loss, same CryptoBench POS_WEIGHT)
  stack3_2stage   per seed, stage 1 trains on the extra chains alone for a fixed 12 epochs from the Phase 266b checkpoint (no
                  CryptoBench chain involved, so one stage-1 model serves all four folds without leakage); stage 2 is the usual
                  deployment fit on the fold's CryptoBench chains starting from the stage-1 weights
CryptoBank labels are noisier than CryptoBench's (Phase 308: median Jaccard .39 on 194 shared chains), so 2stage is expected to be
the safer use; both are tested.
PRE-REGISTERED: PRIMARY = pooled held-out PR-AUC over the 760 CV chains, each arm vs stack3, paired by seed, t AND Wilcoxon < .05 after
Holm over the two. Secondary: per-protein AUC/AP overall, on the cryptic-enriched chains (Phase 300) and on the rest.
usage: build_nb309.py <output_root>"""
import json, subprocess, sys, tempfile
from pathlib import Path
ROOT_OUT = Path(sys.argv[1]).resolve(); SLUG = "309"
HERE = Path(__file__).resolve().parent
tmp = Path(tempfile.mkdtemp())
subprocess.run([sys.executable, str(HERE.parent / "phase306" / "build_nb306.py"), str(tmp)], check=True, capture_output=True)
nb = json.load(open(tmp / "306" / "phase306_deploy_cv.ipynb")); cells = nb["cells"]
src = lambda i: cells[i]["source"] if isinstance(cells[i]["source"], str) else "".join(cells[i]["source"])
def rep(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)
find = lambda key: next(i for i in range(len(cells)) if key in src(i))
cells[0]["source"] = "# Phase 309 - homology-clean CryptoBank chains as extra training data (mixed vs two-stage), deployment-matched CV\n"

i = find("# ---- Ankh-large ----"); s = src(i)       # unused here (as in Phase 307)
a, b = s.index("# ---- Ankh-large ----"), s.index("# ---- Foldseek 3Di + SaProt ----"); s = s[:a] + s[b:]
s = s[:s.index("# ---- holo state")] + "for d in dataset: del d.a3di\n"
cells[i]["source"] = s

i = find("fold_of = {pid: int(k.split"); cells[i]["source"] = src(i) + r'''

# ---- Phase 308d extra chains: never held out (fold -1), never in an early-stopping set ----
XB = torch.load(locate("phase308d_cryptobank_graphs.pt"), map_location="cpu", weights_only=False)
assert XB["topo_names"] == blob["topo_names"] and np.allclose(np.asarray(XB["topo_mean"]), np.asarray(blob["topo_mean"]))
EXTRA = XB["dataset"]; del XB
for d in EXTRA:
    assert d.x.size(1) == N_PRE + TOPO_DIM
    d.x = torch.cat([d.x, torch.zeros(d.num_nodes, GEOM_DIM)], dim=1)   # geometry columns are masked to zero for every chain below
    d.fold, d.split, d.y_main = -1, "extra", d.y.clone()
with ThreadPoolExecutor(max_workers=16) as ex: paths.update({p: f for p, f in ex.map(fetch, sorted({str(d.pdb_id) for d in EXTRA})) if f})
EXTRA = [d for d in EXTRA if str(d.pdb_id) in paths]
for d in EXTRA[:50]: assert len(resnums_of(str(d.pdb_id), str(d.chain_id))) == d.num_nodes
dataset.extend(EXTRA)
print(f"extra CryptoBank chains: {len(EXTRA)}, residues {sum(d.num_nodes for d in EXTRA):,}, positives {sum(int(d.y.sum()) for d in EXTRA):,}")
'''

i = find("ARMS = [("); t = src(i)
t = rep(t, '''    missing, unexpected = model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE), strict=False)''',
        '''    init = pretrained_path if isinstance(pretrained_path, dict) else torch.load(pretrained_path, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(init, strict=False)''')
a, b = t.index("ARMS = [("), t.index("T0 = time.time()")
t = t[:a] + '''S3 = dict(plm_dim=1280, plm2_dim=1024, plm4_dim=1280)
ARMS = [("stack3", S3, {}), ("stack3_mix", S3, {}), ("stack3_2stage", S3, {})]
STAGE1 = {}
def stage1(seed, ck):
    if seed not in STAGE1:
        m = fit_fixed(EXTRA, seed, ck, 1280, "bce", 1.0, epochs=12, plm2_dim=1024, plm4_dim=1280)
        STAGE1[seed] = {k: v.detach().clone() for k, v in m.state_dict().items()}; del m
        print(f"    stage 1 (extra chains only, 12 epochs) seed {seed}: held-out-free PR on all 760 CV chains {pooled_pr_sd(STAGE1[seed]):.4f}", flush=True)
    return STAGE1[seed]
def pooled_pr_sd(sd):
    m = LocalizingEGNN(N_PRE, SIDE_DIM, use_loc_head=True, **S3).to(DEVICE); m.load_state_dict(sd); m.eval()
    return pooled_pr(m, CV)
''' + t[b:]
t = rep(t, "model, ep = fit_es(train_set, es_set, seed, ck, dims, flags)",
        '''init = stage1(seed, ck) if name == "stack3_2stage" else ck
            tr = train_set + EXTRA if name == "stack3_mix" else train_set
            model, ep = fit_es(tr, es_set, seed, init, dims, flags)''')
t = rep(t, 'WORK / "phase306_cv_preds.pt"', 'WORK / "phase309_cv_preds.pt"')
cells[i]["source"] = t

d = ROOT_OUT / SLUG; d.mkdir(parents=True, exist_ok=True)
o = d / "phase309_cryptobank_extra_cv.ipynb"; o.write_text(json.dumps(nb, indent=1) + "\n")
meta = json.load(open(tmp / "306" / "kernel-metadata.json"))
meta.update(id="blackpearls1/phase309-cryptobank-extra-cv", title="Phase309 Cryptobank Extra Cv", code_file=o.name)
meta["dataset_sources"] = [x for x in meta.get("dataset_sources", []) if "hotspot" not in x]
meta["kernel_sources"] = meta["kernel_sources"] + ["blackpearls1/phase308d-cryptobank-graphs"]
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n"); print("wrote", o)
