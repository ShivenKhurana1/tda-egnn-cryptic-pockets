#!/usr/bin/env python3
"""Phase 304: confirmation of the Phase 303 lead on FRESH seeds, with the primary fixed in advance.
Phase 303 (primary pooled PR, not confirmed) showed hot_studemb raising per-protein AUC +0.011 in 8/8 seeds and studemb alone +0.010 on
non-cryptic chains (replicating Phase 301). Same kernel as Phase 303 with arms stack3, stack3_studemb, stack3_hot_studemb and fine-tuning
seeds 8-15 (seed s fine-tunes from pretrained checkpoint s-8: new initialisations of the added heads, new shuffles and dropout; the
pretrained trunks are shared with Phase 303, so the replication is partial, stated as such).
PRE-REGISTERED: PRIMARY = per-protein AUC over all 760 CV chains (mean over proteins, per seed), stack3_studemb and stack3_hot_studemb each
vs stack3, paired by seed, t AND Wilcoxon < .05 after Holm over the two. Secondary: pooled held-out PR, per-protein AP/P@20, cryptic-enriched
and rest AUC.
usage: build_nb304.py <output_root>"""
import json, subprocess, sys, tempfile
from pathlib import Path
ROOT_OUT = Path(sys.argv[1]).resolve(); SLUG = "304"
HERE = Path(__file__).resolve().parent
tmp = Path(tempfile.mkdtemp())
subprocess.run([sys.executable, str(HERE.parent / "phase303" / "build_nb303.py"), str(tmp)], check=True, capture_output=True)
nb = json.load(open(tmp / "303" / "phase303_hot_studemb_cv.ipynb")); cells = nb["cells"]
src = lambda i: cells[i]["source"] if isinstance(cells[i]["source"], str) else "".join(cells[i]["source"])
def rep(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)
i = next(i for i in range(len(cells)) if "ARMS = [(" in src(i))
s = src(i)
s = rep(s, '''ARMS = [("stack3", {}), ("stack3_hot", dict(hot=True)), ("stack3_studemb", dict(auxemb=True)), ("stack3_hot_studemb", dict(hot=True, auxemb=True))]''',
        '''ARMS = [("stack3", {}), ("stack3_studemb", dict(auxemb=True)), ("stack3_hot_studemb", dict(hot=True, auxemb=True))]''')
s = rep(s, "    for seed in range(8):\n", "    for seed in range(8, 16):\n")
s = rep(s, 'ck = locate(f"pretrain266_full_seed{seed}.pt")', 'ck = locate(f"pretrain266_full_seed{seed - 8}.pt")')
s = rep(s, 'WORK / "phase303_cv_preds.pt"', 'WORK / "phase304_cv_preds.pt"')
cells[i]["source"] = s
cells[0]["source"] = "# Phase 304 - confirmation of studemb / hot_studemb on fresh fine-tuning seeds 8-15, per-protein AUC primary\n"
d = ROOT_OUT / SLUG; d.mkdir(parents=True, exist_ok=True)
o = d / "phase304_confirm_cv.ipynb"; o.write_text(json.dumps(nb, indent=1) + "\n")
meta = json.load(open(tmp / "303" / "kernel-metadata.json")); meta.update(id="blackpearls1/phase304-confirm-cv", title="Phase304 Confirm Cv", code_file=o.name)
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n"); print("wrote", o)
