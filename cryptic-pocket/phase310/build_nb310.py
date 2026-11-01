#!/usr/bin/env python3
"""Phase 310: 16-seed stack3 test ensemble, Phase 299/305 protocol (early stopping on the 60 val chains, official union labels).
Only 8 Phase 266b pretrained checkpoints exist, so fine-tune seed s starts from checkpoint s % 8: seeds 0-7 repeat Phase 305's stack3,
seeds 8-15 reuse the same checkpoints with a different fine-tune seed (data order, dropout, inner init of the new branches).
PRE-REGISTERED: PRIMARY = 16-seed ensemble vs the 8-seed ensemble of seeds 0-7 on test pooled PR and ROC, 95% bootstrap CI over test
proteins (paired resampling) excluding 0. Secondary: MCC (val-tuned threshold), per-protein AP/AUC, P@20, strictly-cryptic AUC, the
1..16 seed curve, and 16 vs the mean of random 8-of-16 subsets (so seeds 0-7 are not a lucky draw).
Unused Ankh, holo-state and hotspot cells inherited from Phase 303/305 are stripped.
usage: build_nb310.py <output_root>"""
import json, subprocess, sys, tempfile
from pathlib import Path
ROOT_OUT = Path(sys.argv[1]).resolve(); SLUG = "310"
HERE = Path(__file__).resolve().parent
tmp = Path(tempfile.mkdtemp())
subprocess.run([sys.executable, str(HERE.parent / "phase305" / "build_nb305.py"), str(tmp)], check=True, capture_output=True)
nb = json.load(open(tmp / "305" / "phase305_stack3hs_test.ipynb")); cells = nb["cells"]
src = lambda i: cells[i]["source"] if isinstance(cells[i]["source"], str) else "".join(cells[i]["source"])
def rep(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)
find = lambda key: next(i for i in range(len(cells)) if key in src(i))
cells[0]["source"] = "# Phase 310 - 16-seed stack3 test ensemble (Phase 305 protocol)\n"

i = find("# ---- Ankh-large ----"); s = src(i)
a, b = s.index("# ---- Ankh-large ----"), s.index("# ---- Foldseek 3Di + SaProt ----"); s = s[:a] + s[b:]
s = s[:s.index("# ---- holo state")] + "for d in dataset: del d.a3di\n"
cells[i]["source"] = s

i = find("ARMS = [("); t = src(i)
t = rep(t, 'ARMS = [("stack3", {}), ("stack3_hot_studemb", dict(hot=True, auxemb=True))]', 'ARMS = [("stack3", {})]')
t = rep(t, "for seed in range(8):", "for seed in range(16):")
t = rep(t, 'locate(f"pretrain266_full_seed{seed}.pt")', 'locate(f"pretrain266_full_seed{seed % 8}.pt")')
t = t.replace('WORK / "phase305_test_preds.pt"', 'WORK / "phase310_test_preds.pt"')
cells[i]["source"] = t

d = ROOT_OUT / SLUG; d.mkdir(parents=True, exist_ok=True)
o = d / "phase310_stack3_16seed_test.ipynb"; o.write_text(json.dumps(nb, indent=1) + "\n")
meta = json.load(open(tmp / "305" / "kernel-metadata.json"))
meta.update(id="blackpearls1/phase310-stack3-16seed-test", title="Phase310 Stack3 16seed Test", code_file=o.name)
meta["dataset_sources"] = [x for x in meta.get("dataset_sources", []) if "hotspot" not in x]
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n"); print("wrote", o)
