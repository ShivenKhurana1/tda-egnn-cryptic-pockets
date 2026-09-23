#!/usr/bin/env python3
"""Phase 251: the COACH420-clean scale-3 training list (Phase 53's 7,181 chains) with recovered true ligand centers,
saved once so the four training kernels (Phase 253a/b) and the scorer share an identical, fixed training set.
Splices Phase 53's own cells verbatim (setup, COACH420 fetch, MMseqs2 leakage exclusion, true-center recovery)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
nb53 = json.load(open(ROOT / "phase53_scaled3_coach420.ipynb"))
s53 = lambda i: "".join(nb53["cells"][i]["source"])

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r'''
# Phase 251 — COACH420-clean scale-3 training list with true ligand centers

Phase 53/56 trained on the 7,513-chain round-3 pool after excluding every chain homologous (>=30% identity, MMseqs2)
to COACH420 (7,181 chains kept), with each chain's true ligand center recovered from its structure. This notebook reruns
exactly those steps (Phase 53's cells, verbatim) and saves the result -- the ordered key list and the true centers -- so the
geometry-control training kernels all start from one fixed, identical training set instead of each redoing the MMseqs2 run.
''')
md("## Cell 1 — Install")
code("!pip install -q torch-geometric==2.6.1 gudhi==3.10.1 biopython==1.84 transformers==4.46.3")
md("## Cell 2 — Setup (Phase 53 cell; the unused Phase 51 topology blob is not loaded)")
code(s53(3).replace('blob6b = torch.load(locate("phase51_graphs.pt"), map_location="cpu", weights_only=False)\n', '')
           .replace('TOPO_NAMES = blob6b["topo_names"]\n', ''))
md("## Cell 3 — Fetch ALL 420 COACH420 structures (Phase 53, verbatim)")
code(s53(5))
md("## Cell 4 — Clean the TRAINING side: exclude any chain homologous to COACH420 (Phase 53, verbatim)")
code(s53(7))
md("## Cell 5 — Recover true ligand centers for the cleaned training set (Phase 53, verbatim)")
code(s53(9))
md("## Cell 6 — Save the fixed training list and centers")
code(r'''
order_keys = [(str(d.pdb_id), str(d.chain_id)) for d in dataset]          # dataset order after all filters
centers = {(str(d.pdb_id), str(d.chain_id)): d.true_center.numpy().astype(np.float32) for d in dataset}
assert len(order_keys) == len(set(order_keys)) == len(centers)
torch.save(dict(order_keys=order_keys, centers=centers, n_train_excluded=len(leaked_train_keys),
                n_clean=len(order_keys)), WORK / "phase251_clean_centers.pt")
print(f"saved {len(order_keys):,} chains with true centers -> phase251_clean_centers.pt "
      f"({len(leaked_train_keys)} excluded as homologous to COACH420)")
assert 7000 <= len(order_keys) <= 7300, f"expected about 7,181 chains (Phase 53), got {len(order_keys)}"
''')
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
o = HERE / "phase251_clean_centers.ipynb"
o.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
meta = {"id": "blackpearls1/phase251-clean-centers", "title": "Phase251 Clean Centers", "code_file": o.name,
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False,
        "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["blackpearls1/tda-egnn-phase52-esm-dihedral-scale3"]}
(HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
