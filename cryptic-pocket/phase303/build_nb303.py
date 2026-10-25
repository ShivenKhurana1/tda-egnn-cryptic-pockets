#!/usr/bin/env python3
"""Phase 303: two levers that helped FORMED pockets, built into the network, on the Phase 295 CV instrument (all arms in one kernel).
  hot      Phase 302 pseudo-FTMap hotspot features (11 per residue; Kaggle dataset blackpearls1/phase302-hotspot-features, keyed
           "pdb_chain" in node order) through a zero-initialised branch into the EGNN's first hidden state; standardised over all chains,
           clamped to +-5; chains without features get zeros. (Phase 302: stacking them on stack3 gave +0.010 AUC on non-cryptic chains.)
  studemb  Phase 301 auxiliary head regressing the holo-state SaProt embedding (+0.010 AUC on non-cryptic chains, 8/8).
Arms: stack3 (comparator), stack3_hot, stack3_studemb, stack3_hot_studemb; 8 seeds x 4 folds, fixed 12 epochs.
PRE-REGISTERED: PRIMARY = pooled held-out PR-AUC over the 760 CV chains, each arm vs stack3, paired by seed, t AND Wilcoxon < .05 after
Holm over the three. Secondary: per-protein AUC/AP/P@20 overall, on the cryptic-enriched chains and on the rest.
usage: build_nb303.py <output_root>"""
import json, subprocess, sys, tempfile
from pathlib import Path

ROOT_OUT = Path(sys.argv[1]).resolve(); SLUG = "303"
HERE = Path(__file__).resolve().parent
tmp = Path(tempfile.mkdtemp())
subprocess.run([sys.executable, str(HERE.parent / "phase301" / "build_nb301.py"), str(tmp)], check=True, capture_output=True)
nb = json.load(open(tmp / "301" / "phase301_holo_privileged_cv.ipynb"))
cells = nb["cells"]
src = lambda i: cells[i]["source"] if isinstance(cells[i]["source"], str) else "".join(cells[i]["source"])
def setsrc(i, s): cells[i]["source"] = s
def rep(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)
find = lambda key: next(i for i in range(len(cells)) if key in src(i))

setsrc(0, "# Phase 303 - hotspot-feature branch and holo-embedding student built into stack3, Phase 295 CV instrument\n")
i = find("holo-state SaProt embeddings [")
setsrc(i, src(i) + r'''

HZ = np.load(locate("hotspot_feats.npz")); have = set(HZ.files)
allF = np.vstack([HZ[k] for k in HZ.files]); mu, sd = allF.mean(0), allF.std(0) + 1e-6
n_hot = 0
for d in dataset:
    k = f"{d.pdb_id}_{d.chain_id}"
    if k in have and HZ[k].shape[0] == d.num_nodes:
        d.hot = torch.from_numpy(np.clip((HZ[k] - mu) / sd, -5, 5)).float(); n_hot += 1
    else:
        d.hot = torch.zeros(d.num_nodes, 11)
print(f"hotspot features attached to {n_hot}/{len(dataset)} chains")
''')
i = find("class EGCL"); m = src(i)
m = rep(m, "oracle=False, aux3di=False, auxemb=False):", "oracle=False, aux3di=False, auxemb=False, hot=False):")
m = rep(m, "        if plm4_dim:\n            s.plm4_proj", "        s.hot = hot\n        if hot:\n            s.hot_proj=nn.Linear(11,h); nn.init.zeros_(s.hot_proj.weight); nn.init.zeros_(s.hot_proj.bias)\n        if plm4_dim:\n            s.plm4_proj")
m = rep(m, "        if s.plm4_dim: h=h+s.plm4_proj((d.plm5_emb if s.oracle else d.plm4_emb).float())\n",
        "        if s.plm4_dim: h=h+s.plm4_proj((d.plm5_emb if s.oracle else d.plm4_emb).float())\n        if s.hot: h=h+s.hot_proj(d.hot)\n")
setsrc(i, m)
i = find("def run_epoch"); t = src(i)
t = rep(t, '"h3_head", "hemb_head"))', '"h3_head", "hemb_head", "hot_proj"))')
setsrc(i, t)
i = find('ARMS = [("stack3", {})')
setsrc(i, rep(src(i), '''ARMS = [("stack3", {}), ("oracle_holo", dict(oracle=True)), ("stud3di", dict(aux3di=True)), ("studemb", dict(auxemb=True))]''',
              '''ARMS = [("stack3", {}), ("stack3_hot", dict(hot=True)), ("stack3_studemb", dict(auxemb=True)), ("stack3_hot_studemb", dict(hot=True, auxemb=True))]'''))
i = find('ARMS = [("stack3", {})'); setsrc(i, rep(src(i), 'WORK / "phase301_cv_preds.pt"', 'WORK / "phase303_cv_preds.pt"'))
d = ROOT_OUT / SLUG; d.mkdir(parents=True, exist_ok=True)
o = d / "phase303_hot_studemb_cv.ipynb"; o.write_text(json.dumps(nb, indent=1) + "\n")
meta = json.load(open(tmp / "301" / "kernel-metadata.json"))
meta.update(id="blackpearls1/phase303-hot-studemb-cv", title="Phase303 Hot Studemb Cv", code_file=o.name, dataset_sources=["blackpearls1/phase302-hotspot-features"])
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote", o)
