#!/usr/bin/env python3
"""Chunked 650M-ESM embedding for the main pipeline's round-3 topology
pool (Phase51, 7,513 chains) -- building a 650M main-pipeline checkpoint
to unlock CryptoBench transfer learning at that scale too (currently the
only scale still stuck on from-scratch). Live 650M CPU inference over the
FULL pool would likely take ~18hrs (extrapolated from CryptoBench's own
35M CPU timing, scaled by param count and residue count) -- too long for
one 12hr session, so this chunks the pool (mirroring the proven HOLO4K
featurization chunking pattern, Phase108-110) into N_CHUNKS pieces, each
computing 650M embeddings for its slice only. Uses the SAME fp16-storage
safety pattern proven in Phase96 (3B blob) even though 650M is smaller,
since it has zero downside and avoids re-discovering the same RAM/disk
issues. Usage: python3 build_nb_650m_chunk.py <CHUNK_IDX> <N_CHUNKS> <PHASE_NUM>"""
import json, pickle, sys
from pathlib import Path

CHUNK_IDX = int(sys.argv[1])
N_CHUNKS = int(sys.argv[2])
PHASE_NUM = sys.argv[3]

C10 = pickle.load(open("/tmp/p10_cells.pkl", "rb"))
s10 = lambda i: (C10[i]["source"] if isinstance(C10[i]["source"], str) else "".join(C10[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(f'''
# Phase {PHASE_NUM} - Main Pipeline 650M-ESM Chunk {CHUNK_IDX}/{N_CHUNKS}

Building a 650M-ESM main-pipeline checkpoint (round-3 COACH420 pool,
Phase51) to unlock CryptoBench transfer learning at 650M scale -- the
last scale still stuck on from-scratch (no matching main-pipeline
checkpoint existed at 650M until now). Chunked ({N_CHUNKS} pieces) since
full-pool 650M CPU embedding would likely exceed the 12hr session cap.
This notebook processes chunk {CHUNK_IDX} only.
''')

md(r'''## Cell 0 - Install dependencies''')
code(s10(2))

md(f'''## Cell 1 - Setup, load Phase 51's topology-fused pool, slice to chunk {CHUNK_IDX}/{N_CHUNKS}''')
code(s10(3).replace('locate("phase6b_graphs.pt")', 'locate("phase51_graphs.pt")')
           .replace('dataset = blob["dataset"]',
                     f'dataset = blob["dataset"]\n'
                     f'_n = len(dataset); _lo = ({CHUNK_IDX} * _n) // {N_CHUNKS}; _hi = (({CHUNK_IDX}+1) * _n) // {N_CHUNKS}\n'
                     f'dataset = dataset[_lo:_hi]\n'
                     f'print(f"chunk {CHUNK_IDX}/{N_CHUNKS}: chains {{_lo}}:{{_hi}} ({{len(dataset)}} of {{_n}} total)")'))

md(r'''## Cell 2 - Recompute real backbone dihedrals (identical to Phase 10/30/37/52/92/95/96)''')
code(s10(5))

md(r'''## Cell 3 - ESM2-650M embeddings (fp16 storage -- proven pattern from Phase96''' + "'" + r'''s 3B postmortem, applied proactively here)''')
code(r'''
import gc
from transformers import AutoTokenizer, AutoModel

ESM_NAME = "facebook/esm2_t33_650M_UR50D"   # 650M params, hidden 1280
tok = AutoTokenizer.from_pretrained(ESM_NAME)
esm = AutoModel.from_pretrained(ESM_NAME).half().to(DEVICE).eval()
ESM_DIM = esm.config.hidden_size
print(f"{ESM_NAME}: hidden {ESM_DIM}, device {DEVICE}, dtype {next(esm.parameters()).dtype}")

@torch.no_grad()
def embed_batch(seqs, max_len=1002):
    enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    out = esm(**enc).last_hidden_state
    embs = []
    for i, s in enumerate(seqs):
        n = len(s)
        embs.append(out[i, 1:1+n, :].cpu())
    return embs

order = sorted(range(len(dataset)), key=lambda i: len(dataset[i].seq))
BATCH = 2   # 650M is smaller than 3B; still conservative given CPU
t0=time.time()
esm_feats = [None]*len(dataset)
for bi in range(0, len(order), BATCH):
    idx = order[bi:bi+BATCH]
    seqs = [dataset[i].seq for i in idx]
    embs = embed_batch(seqs)
    for i, e in zip(idx, embs):
        esm_feats[i] = e
    if bi % (BATCH*100) == 0:
        print(f"  {bi}/{len(order)}  [{(time.time()-t0)/60:.1f}m]")

n_res = sum(int(d.num_nodes) for d in dataset)
esm_gb = sum(e.numel()*e.element_size() for e in esm_feats if e is not None) / 1e9
print(f"\nESM2 embeddings for {len(dataset):,} chains / {n_res:,} residues "
      f"in {(time.time()-t0)/60:.1f} min ({1000*(time.time()-t0)/n_res:.2f} ms/residue)")
print(f"esm_feats resident size: {esm_gb:.1f} GB (fp16)")
del esm; torch.cuda.empty_cache(); gc.collect()
''')

md(f'''## Cell 4 - Assemble augmented features for this chunk, save AS FP16''')
code(f'''
kept, dropped = [], 0
for i, d in enumerate(dataset):
    if dihed[i] is None or esm_feats[i] is None:
        dropped += 1; continue
    n = int(d.num_nodes)
    if dihed[i].shape[0] != n or esm_feats[i].shape[0] != n:
        dropped += 1; continue
    onehot = d.x[:, :21]
    topo_z = d.x[:, N_SEQ:]
    d.x = torch.cat([onehot, dihed[i], esm_feats[i].float(), topo_z], dim=1).half()
    esm_feats[i] = None
    kept.append(d)
    if i % 500 == 0:
        gc.collect()

del esm_feats; gc.collect()
print(f"kept {{len(kept):,}}/{{len(dataset):,}} chains (dropped {{dropped}}: parse/alignment failures)")
N_PRE = 21 + 6 + ESM_DIM
print(f"chunk node feature dim: {{kept[0].x.size(1)}}  (21 onehot + 6 dihedral + "
      f"{{ESM_DIM}} ESM + {{TOPO_DIM}} topo), stored dtype {{kept[0].x.dtype}}")
for d in kept[:20]:
    assert torch.isfinite(d.x.float()).all(), f"{{d.pdb_id}}_{{d.chain_id}}: non-finite features"
print("finite-value spot check: OK")

torch.save({{"dataset": kept, "n_pre_topo": N_PRE, "topo_dim": TOPO_DIM,
            "esm_dim": ESM_DIM, "pos_weight": blob["pos_weight"]}},
           "/kaggle/working/phase{PHASE_NUM}_650m_chunk{CHUNK_IDX}.pt")
import os
sz = os.path.getsize("/kaggle/working/phase{PHASE_NUM}_650m_chunk{CHUNK_IDX}.pt") / 1e9
print(f"saved -> /kaggle/working/phase{PHASE_NUM}_650m_chunk{CHUNK_IDX}.pt ({{sz:.2f}} GB on disk)")
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path(f"/Users/shivenkhurana/bioproj/phase{PHASE_NUM}_650m_chunk{CHUNK_IDX}.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
