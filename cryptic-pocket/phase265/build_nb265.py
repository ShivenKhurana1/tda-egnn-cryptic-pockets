#!/usr/bin/env python3
"""Phase 265: how many CryptoBench chains have a >=30%-identity homolog in the main-pipeline pretraining pool (the pool that seeds the 3B transfer)?
Phase 97/263 cleaned the pool only against COACH420; the CryptoBench transfer numbers (Phase 129/264) never checked it against CryptoBench.
CPU kernel. Pool = Phase 251's clean list (chain sequences from the Phase 96 blob); CryptoBench = Phase 48 blob (sequences decoded losslessly from the
one-hot block). MMseqs2 easy-cluster at 30% identity, 80% coverage (cov-mode 1), exactly as Phase 97."""
import json
from pathlib import Path
C = []
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})
code("!pip install -q torch-geometric==2.6.1")
code(r'''
import json, os, shutil, subprocess
from collections import defaultdict
from pathlib import Path
import torch

def locate(f):
    c=[]
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): c.extend(r.rglob(f))
    if not c: raise FileNotFoundError(f)
    return max(c, key=lambda p: p.stat().st_size)

AA = "ACDEFGHIKLMNPQRSTVWYX"
clean = torch.load(locate("phase251_clean_centers.pt"), map_location="cpu", weights_only=False)
keys = set(clean["order_keys"])
blob = torch.load(locate("phase96_graphs_esm3b.pt"), map_location="cpu", weights_only=False)
pool = {(str(d.pdb_id), str(d.chain_id)): d.seq for d in blob["dataset"] if (str(d.pdb_id), str(d.chain_id)) in keys}
del blob
print("pool chains with sequence:", len(pool), "of", len(keys))
cb = torch.load(locate("phase48_cryptobench_graphs_esm3b.pt"), map_location="cpu", weights_only=False)["dataset"]
cbseq = {}
for d in cb:
    idx = d.x[:, :21].float().argmax(1); assert bool((d.x[:, :21].float().sum(1) == 1).all())
    cbseq[(str(d.pdb_id), str(d.chain_id), str(d.split))] = "".join(AA[i] for i in idx.tolist())
print("CryptoBench chains:", len(cbseq))
del cb

subprocess.run("curl -sL https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -o /tmp/mm.tar.gz && tar xzf /tmp/mm.tar.gz -C /tmp", shell=True, check=True, timeout=600)
mm = "/tmp/mmseqs/bin/mmseqs"; os.chmod(mm, 0o755)
W = Path("/tmp/cbleak"); shutil.rmtree(W, ignore_errors=True); W.mkdir()
lines = [f">POOL__{k[0]}_{k[1]}\n{s}\n" for k, s in pool.items()] + [f">CB__{k[0]}_{k[1]}__{k[2]}\n{s}\n" for k, s in cbseq.items()]
(W / "all.fasta").write_text("".join(lines))
subprocess.run([mm, "easy-cluster", str(W/"all.fasta"), str(W/"res"), str(W/"tmp"), "--min-seq-id", "0.30", "-c", "0.8", "--cov-mode", "1", "-v", "1"],
               check=True, capture_output=True, timeout=3600)
rep = defaultdict(list)
for line in (W / "res_cluster.tsv").read_text().splitlines():
    a, b = line.split("\t"); rep[a].append(b)
for r_, members in rep.items():
    if r_ not in members: members.append(r_)
leaked = {}
for members in rep.values():
    if any(m.startswith("POOL__") for m in members):
        for m in members:
            if m.startswith("CB__"): leaked[m] = True
out = {"n_pool": len(pool), "n_cb": len(cbseq), "cb": {}}
for k in cbseq:
    out["cb"][f"{k[0]}_{k[1]}"] = {"split": k[2], "leaked": f"CB__{k[0]}_{k[1]}__{k[2]}" in leaked}
for sp in ("train", "val", "test"):
    v = [x for x in out["cb"].values() if x["split"] == sp]
    print(f"{sp}: {sum(x['leaked'] for x in v)}/{len(v)} CryptoBench chains have a >=30% identity homolog in the pretraining pool")
json.dump(out, open("/kaggle/working/phase265_cryptobench_pool_homology.json", "w"), indent=1)
''')
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
d = Path(__file__).resolve().parent / "cpu"; d.mkdir(exist_ok=True)
(d / "phase265_cb_pool_homology.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
(d / "kernel-metadata.json").write_text(json.dumps({"id": "blackpearls1/phase265-cb-pool-homology", "title": "Phase265 Cb Pool Homology", "code_file": "phase265_cb_pool_homology.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
    "dataset_sources": [], "competition_sources": [],
    "kernel_sources": ["blackpearls1/tda-egnn-phase96-esm-dihedral-esm3b-v3", "blackpearls1/phase251-clean-centers", "blackpearls1/tda-egnn-phase48-cryptobench-esm3b"]}, indent=2) + "\n")
print("wrote", d)
