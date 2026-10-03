#!/usr/bin/env python3
"""Phase 96: retry #2 of the ESM2-3B main-pipeline blob. Phase95 fixed the
RAM problem (esm_feats fp16, freed per-chain) and got all the way through
assembly ("kept 7,513/7,513 chains") -- but then torch.save() hit
`OSError: [Errno 28] No space left on device`. Root cause: this is a DISK
QUOTA problem, not RAM -- the final blob at fp32 (2621 dims x 2,211,045
residues x 4 bytes = ~23.6GB) exceeds Kaggle's /kaggle/working quota
(~19.5-20GB), confirmed by the download of Phase95's corrupt partial
output stalling at ~20-21GB total. Fix: save the final per-chain
node-feature tensor in fp16 (~11.8GB, half the size) instead of fp32.
Downstream training notebooks (Phase93/94, both updated) upcast back to
fp32 immediately after loading -- a cheap one-time op on kernels that
don't also hold a 3B-param LM in memory, so training numerics are
unaffected."""
import json, pickle
from pathlib import Path

C10 = pickle.load(open("/tmp/p10_cells.pkl", "rb"))
s10 = lambda i: (C10[i]["source"] if isinstance(C10[i]["source"], str) else "".join(C10[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 96 - Main Pipeline with ESM2-3B, Disk-Quota-Safe Retry

Phase95 fixed Phase92's RAM problem and got all the way through assembly
(kept 7,513/7,513 chains, node dim 2621) -- but `torch.save()` then hit
`OSError: [Errno 28] No space left on device`. The fp32 output (~23.6GB)
exceeds Kaggle's `/kaggle/working` quota (~19.5-20GB). This retry saves
the final blob in fp16 (~11.8GB) -- Phase93/94 upcast to fp32 right after
loading, so training itself still runs at full fp32 precision, only the
on-disk artifact is compressed. Everything else identical to Phase95.
''')

md(r'''## Cell 0 - Install dependencies''')
code(s10(2))

md(r'''## Cell 1 - Setup, load Phase 51's topology-fused pool''')
code(s10(3).replace('locate("phase6b_graphs.pt")', 'locate("phase51_graphs.pt")'))

md(r'''## Cell 2 - Recompute real backbone dihedrals (identical to Phase 10/30/37/52/92/95)''')
code(s10(5))

md(r'''## Cell 3 - ESM2-3B embeddings (fp16 storage -- see Phase92 postmortem)''')
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

md(r'''## Cell 4 - Assemble augmented features, verify alignment, save AS FP16 (disk-quota fix -- everything else identical to Phase95)''')
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
    d.x = torch.cat([onehot, dihed[i], esm_feats[i].float(), topo_z], dim=1).half()  # fp16 on disk
    esm_feats[i] = None                      # free the fp16 tensor now that it's folded in
    kept.append(d)
    if i % 1500 == 0:
        gc.collect()

del esm_feats; gc.collect()
print(f"kept {len(kept):,}/{len(dataset):,} chains (dropped {dropped}: parse/alignment failures)")
N_PRE = 21 + 6 + ESM_DIM       # everything before the topo block
print(f"new node feature dim: {kept[0].x.size(1)}  (21 onehot + 6 dihedral + "
      f"{ESM_DIM} ESM + {TOPO_DIM} topo), stored dtype {kept[0].x.dtype}")
for d in kept[:50]:
    assert torch.isfinite(d.x.float()).all(), f"{d.pdb_id}_{d.chain_id}: non-finite features"
print("finite-value spot check: OK")

for s in ("train","val","test"):
    ds = [d for d in kept if d.split == s]
    print(f"  {s:<6} {len(ds):>5} chains")

torch.save({"dataset": kept, "n_pre_topo": N_PRE, "topo_dim": TOPO_DIM,
            "esm_dim": ESM_DIM, "pos_weight": blob["pos_weight"]},
           "/kaggle/working/phase96_graphs_esm3b.pt")
import os
sz = os.path.getsize("/kaggle/working/phase96_graphs_esm3b.pt") / 1e9
print(f"saved -> /kaggle/working/phase96_graphs_esm3b.pt ({sz:.1f} GB on disk)")
''')

md(r'''
---

`phase96_graphs_esm3b.pt` -- same content as Phase95 was trying to
produce (node dim 2621 = 21+6+2560+34) but **stored as fp16, not fp32**.
Any notebook loading this MUST upcast: `for d in dataset: d.x = d.x.float()`
right after `dataset = blob["dataset"]`, before anything else touches it
(Phase93/94 both updated to do this). N_PRE/TOPO_DIM are read dynamically
from the blob either way.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase96_esm_dihedral_esm3b_v3.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
