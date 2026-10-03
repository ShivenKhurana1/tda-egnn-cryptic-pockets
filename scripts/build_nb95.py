#!/usr/bin/env python3
"""Phase 95: retry of Phase92 (ESM2-3B main-pipeline blob), which reached
'ESM2 embeddings for 7,513 chains ... done' and then died with a silent
DeadKernelError ~10s later (no Python traceback -- classic host-RAM OOM
kill, not a CUDA OOM). Diagnosis: Phase92 kept esm_feats as one fp32
tensor per chain (2,211,045 residues x 2560 dims x 4 bytes = ~22.6GB) all
resident at once, on top of the topology dataset and the about-to-be-built
23.6GB final feature blob (2621 dims) -- CryptoBench's identical pattern
(build_nb48.py) never hit this because its total residue count was far
smaller. Fix: keep esm_feats in fp16 (~11.3GB) and free each entry
immediately after it's folded into the final per-chain tensor, so the
fp16 pool shrinks as the fp32 output grows instead of both being resident
at peak."""
import json, pickle
from pathlib import Path

C10 = pickle.load(open("/tmp/p10_cells.pkl", "rb"))
s10 = lambda i: (C10[i]["source"] if isinstance(C10[i]["source"], str) else "".join(C10[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 95 - Main Pipeline with ESM2-3B, Memory-Safe Retry

Phase92 completed the expensive part (3B-model embedding of all 7,513
chains, 16.0 min) then died silently ~10s later with no Python traceback
-- a host-RAM OOM kill, not a CUDA OOM. Root cause: holding the full
esm_feats list as fp32 (~22.6GB) simultaneously with building the final
23.6GB assembled blob. This retry stores embeddings as fp16 (~11.3GB) and
frees each chain's entry the instant it's folded into the final tensor,
so peak RAM stays bounded. Everything else identical to Phase92 (same
Phase51 topology blob, same dihedral logic).
''')

md(r'''## Cell 0 - Install dependencies''')
code(s10(2))

md(r'''## Cell 1 - Setup, load Phase 51's topology-fused pool''')
code(s10(3).replace('locate("phase6b_graphs.pt")', 'locate("phase51_graphs.pt")'))

md(r'''## Cell 2 - Recompute real backbone dihedrals (identical to Phase 10/30/37/52/92)''')
code(s10(5))

md(r'''## Cell 3 - ESM2-3B embeddings (fp16 storage this time -- see Phase92 postmortem above)''')
code(r'''
import gc
from transformers import AutoTokenizer, AutoModel

ESM_NAME = "facebook/esm2_t36_3B_UR50D"   # 3B params, hidden 2560
tok = AutoTokenizer.from_pretrained(ESM_NAME)
esm = AutoModel.from_pretrained(ESM_NAME).half().to(DEVICE).eval()   # fp16: inference-only, no grads needed
ESM_DIM = esm.config.hidden_size
print(f"{ESM_NAME}: hidden {ESM_DIM}, device {DEVICE}, dtype {next(esm.parameters()).dtype}")

@torch.no_grad()
def embed_batch(seqs, max_len=1002):
    enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    out = esm(**enc).last_hidden_state          # (B, L+2, H) incl BOS/EOS, still fp16
    embs = []
    for i, s in enumerate(seqs):
        n = len(s)
        embs.append(out[i, 1:1+n, :].cpu())      # strip BOS, keep exactly n residues -- STAYS fp16
    return embs

order = sorted(range(len(dataset)), key=lambda i: len(dataset[i].seq))
BATCH = 1   # conservative: 3B params, even though seqs are capped shorter than CryptoBench's
t0=time.time()
esm_feats = [None]*len(dataset)
for bi in range(0, len(order), BATCH):
    idx = order[bi:bi+BATCH]
    seqs = [dataset[i].seq for i in idx]
    embs = embed_batch(seqs)
    for i, e in zip(idx, embs):
        esm_feats[i] = e
    if bi % (BATCH*200) == 0:
        print(f"  {bi}/{len(order)}  [{(time.time()-t0)/60:.1f}m]")

n_res = sum(int(d.num_nodes) for d in dataset)
esm_gb = sum(e.numel()*e.element_size() for e in esm_feats if e is not None) / 1e9
print(f"\nESM2 embeddings for {len(dataset):,} chains / {n_res:,} residues "
      f"in {(time.time()-t0)/60:.1f} min ({1000*(time.time()-t0)/n_res:.2f} ms/residue)")
print(f"esm_feats resident size: {esm_gb:.1f} GB (fp16)")
del esm; torch.cuda.empty_cache(); gc.collect()
''')

md(r'''## Cell 4 - Assemble augmented features, verify alignment, save (fp16->fp32 cast + free happen per-chain to cap peak RAM)''')
code(r'''
kept, dropped = [], 0
for i, d in enumerate(dataset):
    if dihed[i] is None or esm_feats[i] is None:
        dropped += 1; continue
    n = int(d.num_nodes)
    if dihed[i].shape[0] != n or esm_feats[i].shape[0] != n:
        dropped += 1; continue
    onehot = d.x[:, :21]                     # first 21 cols of the original block
    topo_z = d.x[:, N_SEQ:]                  # standardized topo block, unchanged
    d.x = torch.cat([onehot, dihed[i], esm_feats[i].float(), topo_z], dim=1)
    esm_feats[i] = None                      # free the fp16 tensor now that it's folded in
    kept.append(d)
    if i % 1500 == 0:
        gc.collect()

del esm_feats; gc.collect()
print(f"kept {len(kept):,}/{len(dataset):,} chains (dropped {dropped}: parse/alignment failures)")
N_PRE = 21 + 6 + ESM_DIM       # everything before the topo block
print(f"new node feature dim: {kept[0].x.size(1)}  (21 onehot + 6 dihedral + "
      f"{ESM_DIM} ESM + {TOPO_DIM} topo)")
for d in kept[:50]:
    assert torch.isfinite(d.x).all(), f"{d.pdb_id}_{d.chain_id}: non-finite features"
print("finite-value spot check: OK")

for s in ("train","val","test"):
    ds = [d for d in kept if d.split == s]
    print(f"  {s:<6} {len(ds):>5} chains")

torch.save({"dataset": kept, "n_pre_topo": N_PRE, "topo_dim": TOPO_DIM,
            "esm_dim": ESM_DIM, "pos_weight": blob["pos_weight"]},
           "/kaggle/working/phase95_graphs_esm3b.pt")
print("saved -> /kaggle/working/phase95_graphs_esm3b.pt")
''')

md(r'''
---

`phase95_graphs_esm3b.pt` -- same schema/content as what Phase92 was
trying to produce (node dim 2621 = 21+6+2560+34). N_PRE/TOPO_DIM are read
dynamically from the blob, so `LocalizingEGNN`/`fit`/`run_epoch` apply
unchanged. Next: point Phase93/94 (COACH420/HOLO4K on the 3B blob) at
`phase95_graphs_esm3b.pt` instead of the failed `phase92_graphs_esm3b.pt`.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase95_esm_dihedral_esm3b_v2.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
