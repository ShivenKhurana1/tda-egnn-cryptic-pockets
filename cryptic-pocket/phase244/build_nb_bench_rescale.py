#!/usr/bin/env python3
"""Re-embed Phase 15's cached COACH420/HOLO4K benchmark features at a different ESM2 scale, WITHOUT
re-fetching or re-parsing any structure. Phase 15's cached x is [onehot(21) | dihedral(6) | ESM-35M(480) |
topo-raw(34)]; onehot and dihedral are ESM-independent, so the amino-acid sequence is recovered exactly by
argmax over the onehot block (a lossless decode -- Phase 15 built it as a clean one-hot), then re-embedded
at the target scale and reassembled. pos/edge_index/edge_attr/ca/ligs are structure-derived and unchanged.

usage: build_nb_bench_rescale.py <650m|3b> <output_dir>
"""
import json, sys
from pathlib import Path

SCALE = sys.argv[1]
OUT_DIR = Path(sys.argv[2]).resolve()
CFG = {
    "650m": dict(esm_name="facebook/esm2_t33_650M_UR50D", batch=4, max_len=1502,
                phase="244", out="phase244_raw_features_esm650m.pt",
                comment="650M params, hidden 1280, fp16, batch 4"),
    "3b": dict(esm_name="facebook/esm2_t36_3B_UR50D", batch=1, max_len=1002,
              phase="245", out="phase245_raw_features_esm3b.pt",
              comment="3B params, hidden 2560, fp16, batch 1"),
}[SCALE]

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(f'''
# Phase {CFG["phase"]} — Re-embed the COACH420/HOLO4K benchmark cache at ESM2-{SCALE.upper()}

Decodes the amino-acid sequence losslessly from Phase 15's cached one-hot block (no re-fetch, no re-parse),
re-embeds every chain at ESM2-{SCALE.upper()} ({CFG["comment"]}), and reassembles
`[onehot(21) | dihedral(6) | ESM-{SCALE.upper()} | topo-raw(34)]` -- a drop-in replacement for
`phase15_raw_features.pkl` at this embedding scale, for the localization geometry control (Phase 246/247).
''')
md("## Cell 1 — Install")
code("!pip install -q biopython==1.84 transformers==4.46.3 accelerate")
md("## Cell 2 — Load Phase 15's cache, decode sequences from the one-hot block")
code(r'''
import pickle, time
from pathlib import Path
import numpy as np
import torch

def locate(f):
    c=[]
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): c.extend(r.rglob(f))
    if not c: raise FileNotFoundError(f)
    return max(c, key=lambda p: p.stat().st_size)

with open(locate("phase15_raw_features.pkl"), "rb") as fh:
    raw_cache = pickle.load(fh)

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"          # identical ordering to Phase 15's own A2I
N_PRE_OLD, TOPO_DIM = 507, 34                   # 21 onehot + 6 dihedral + 480 ESM-35M, then 34 raw topo
assert N_PRE_OLD == 21 + 6 + 480

seqs = {}
for k, e in raw_cache.items():
    oh = e["x"][:, :21]
    idx = oh.argmax(axis=1)
    assert np.allclose(oh[np.arange(len(idx)), idx], 1.0), f"{k}: onehot block is not a clean one-hot"
    seqs[k] = "".join(AA_ALPHABET[i] for i in idx)

print(f"{len(raw_cache):,} cached chains; decoded {len(seqs):,} sequences "
      f"({sum(len(s) for s in seqs.values()):,} residues total)")

GPU_OK=False
if torch.cuda.is_available():
    try:
        _p=torch.randn(256,256,device="cuda"); (_p@_p).sum().item(); torch.cuda.synchronize(); GPU_OK=True
        print(f"GPU: {torch.cuda.get_device_name(0)} (probe PASS)")
    except RuntimeError as e: print("GPU unusable:", e)
DEVICE = torch.device("cuda" if GPU_OK else "cpu")
''')
md(f"## Cell 3 — ESM2-{SCALE.upper()} embeddings for the decoded sequences")
code(f'''
from transformers import AutoTokenizer, AutoModel

ESM_NAME = "{CFG["esm_name"]}"   # {CFG["comment"]}
tok = AutoTokenizer.from_pretrained(ESM_NAME)
esm = AutoModel.from_pretrained(ESM_NAME).half().to(DEVICE).eval()
ESM_DIM = esm.config.hidden_size
print(f"{{ESM_NAME}}: hidden {{ESM_DIM}}, device {{DEVICE}}, dtype {{next(esm.parameters()).dtype}}")

@torch.no_grad()
def embed_batch(chunk_seqs, max_len={CFG["max_len"]}):
    enc = tok(chunk_seqs, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    enc = {{k: v.to(DEVICE) for k, v in enc.items()}}
    out = esm(**enc).last_hidden_state
    embs = []
    for i, s in enumerate(chunk_seqs):
        n = len(s)
        embs.append(out[i, 1:1+n, :].float().cpu().numpy())
    return embs

keys = sorted(seqs, key=lambda k: len(seqs[k]))
BATCH = {CFG["batch"]}
t0 = time.time()
new_esm = {{}}
for bi in range(0, len(keys), BATCH):
    chunk = keys[bi:bi+BATCH]
    embs = embed_batch([seqs[k] for k in chunk])
    for k, e in zip(chunk, embs):
        new_esm[k] = e
    if bi % (BATCH*200) == 0:
        print(f"  {{bi}}/{{len(keys)}}  [{{(time.time()-t0)/60:.1f}}m]")

n_res = sum(len(s) for s in seqs.values())
print(f"\\nembedded {{len(new_esm):,}} chains / {{n_res:,}} residues in {{(time.time()-t0)/60:.1f}} min "
      f"({{1000*(time.time()-t0)/n_res:.2f}} ms/residue)")
del esm; torch.cuda.empty_cache()
''')
md("## Cell 4 — Reassemble and save")
code(r'''
new_cache = {}
n_bad = 0
for k, e in raw_cache.items():
    onehot = e["x"][:, :21]; dihedral = e["x"][:, 21:N_PRE_OLD]; topo = e["x"][:, N_PRE_OLD:]
    esm_e = new_esm[k]
    if esm_e.shape[0] != onehot.shape[0]:
        n_bad += 1; continue
    x_new = np.concatenate([onehot, dihedral, esm_e.astype(np.float32), topo], axis=1)
    if not np.isfinite(x_new).all():
        n_bad += 1; continue
    new_cache[k] = dict(e); new_cache[k]["x"] = x_new

print(f"reassembled {len(new_cache):,}/{len(raw_cache):,} chains ({n_bad} failed alignment/finite check)")
assert n_bad == 0, "some chains failed to reassemble -- investigate before Phase 246/247"
torch.save(dict(cache=new_cache, esm_dim=int(ESM_DIM), n_pre=21+6+int(ESM_DIM), topo_dim=TOPO_DIM),
           "/kaggle/working/__OUT__")
print("saved -> /kaggle/working/__OUT__")
'''.replace("__OUT__", CFG["out"]))

nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
OUT_DIR.mkdir(parents=True, exist_ok=True)
o = OUT_DIR / f"phase{CFG['phase']}_bench_esm{SCALE}.ipynb"
o.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")

meta = {
  "id": f"blackpearls1/phase{CFG['phase']}-bench-esm{SCALE}",
  "title": f"Phase{CFG['phase']} Bench ESM{SCALE.upper()}",
  "code_file": o.name,
  "language": "python", "kernel_type": "notebook", "is_private": True,
  "enable_gpu": True, "machine_shape": "NvidiaTeslaT4", "enable_tpu": False, "enable_internet": True,
  "dataset_sources": [], "competition_sources": [],
  "kernel_sources": ["blackpearls1/tda-egnn-phase15-localized-sota-validation"],
}
(OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"wrote {OUT_DIR/'kernel-metadata.json'}")
