#!/usr/bin/env python3
"""Build a Phase 10-equivalent ingestion notebook for the SAME 2,056-chain pool, but embedded at a
different ESM2 scale (650M or 3B instead of 35M) -- isolates embedding scale from training-pool size,
matching how CryptoBench tested embedding scale (Phase42/45/48: same 942-chain pool, only ESM swapped).

usage: build_nb_embed_scale.py <650m|3b> <output_dir>
"""
import json, sys
from pathlib import Path

SCALE = sys.argv[1]
OUT_DIR = Path(sys.argv[2]).resolve()
ROOT = Path(__file__).resolve().parents[2]
CFG = {
    "650m": dict(esm_name="facebook/esm2_t33_650M_UR50D", batch=4, max_len=1502,
                phase="238", blob_out="phase238_graphs_esm650m.pt",
                comment="650M params, hidden 1280, fp16, batch 4 (T4-safe given up to 1502 tokens/seq)"),
    "3b": dict(esm_name="facebook/esm2_t36_3B_UR50D", batch=1, max_len=1002,
              phase="239", blob_out="phase239_graphs_esm3b.pt",
              comment="3B params, hidden 2560, fp16, batch 1 (conservative on a T4)"),
}[SCALE]

nb10 = json.load(open(ROOT / "phase10_esm_dihedral_boost.ipynb"))
src10 = lambda i: "".join(nb10["cells"][i]["source"])

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(f'''
# Phase {CFG["phase"]} - Same 2,056-chain pool, ESM2-{SCALE.upper()} embeddings

Phase 10 embeds the project's original 2,056-chain homology-clean training pool with ESM2-35M and fuses it
with the existing topology block (Phase 6b). This notebook is identical except the embedding model is
ESM2-{SCALE.upper()} ({CFG["comment"]}) -- everything else (chain list, dihedral computation, topology,
train/val/test split) is unchanged, so the resulting blob is a drop-in replacement for `phase10_graphs.pt`
in the Phase 230/236 geometry-control pipeline, isolating embedding scale as the only variable.
''')
md("## Cell 1 - Install")
code(src10(2).replace("transformers==4.46.3", "transformers==4.46.3 accelerate"))
md("## Cell 2 - Setup, load Phase 6b's topology-fused 2,056-chain pool")
code(src10(3))
md("## Cell 3 - Recompute real backbone dihedrals (identical to Phase 10)")
code(src10(5))
md(f"## Cell 4 - ESM2-{SCALE.upper()} embeddings")
code(f'''
from transformers import AutoTokenizer, AutoModel

ESM_NAME = "{CFG["esm_name"]}"   # {CFG["comment"]}
tok = AutoTokenizer.from_pretrained(ESM_NAME)
esm = AutoModel.from_pretrained(ESM_NAME).half().to(DEVICE).eval()   # fp16: inference-only, no grads needed
ESM_DIM = esm.config.hidden_size
print(f"{{ESM_NAME}}: hidden {{ESM_DIM}}, device {{DEVICE}}, dtype {{next(esm.parameters()).dtype}}")

@torch.no_grad()
def embed_batch(seqs, max_len={CFG["max_len"]}):
    enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    enc = {{k: v.to(DEVICE) for k, v in enc.items()}}
    out = esm(**enc).last_hidden_state
    embs = []
    for i, s in enumerate(seqs):
        n = len(s)
        embs.append(out[i, 1:1+n, :].float().cpu())
    return embs

order = sorted(range(len(dataset)), key=lambda i: len(dataset[i].seq))
BATCH = {CFG["batch"]}
t0=time.time()
esm_feats = [None]*len(dataset)
for bi in range(0, len(order), BATCH):
    idx = order[bi:bi+BATCH]
    seqs = [dataset[i].seq for i in idx]
    embs = embed_batch(seqs)
    for i, e in zip(idx, embs):
        esm_feats[i] = e
    if bi % (BATCH*200) == 0:
        print(f"  {{bi}}/{{len(order)}}  [{{(time.time()-t0)/60:.1f}}m]")

n_res = sum(int(d.num_nodes) for d in dataset)
print(f"\\nESM2 embeddings for {{len(dataset):,}} chains / {{n_res:,}} residues "
      f"in {{(time.time()-t0)/60:.1f}} min ({{1000*(time.time()-t0)/n_res:.2f}} ms/residue)")
del esm; torch.cuda.empty_cache()
''')
md("## Cell 5 - Assemble augmented features and verify alignment")
code(src10(9).replace('"/kaggle/working/phase10_graphs.pt"', f'"/kaggle/working/{CFG["blob_out"]}"')
             .replace('saved -> /kaggle/working/phase10_graphs.pt', f'saved -> /kaggle/working/{CFG["blob_out"]}'))

nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
OUT_DIR.mkdir(parents=True, exist_ok=True)
o = OUT_DIR / f"phase{CFG['phase']}_esm{SCALE}_ingestion.ipynb"
o.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
