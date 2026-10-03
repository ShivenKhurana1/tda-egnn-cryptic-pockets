#!/usr/bin/env python3
"""Phase 92: apply the ESM2 capacity lever (proven on CryptoBench: 35M ->
650M -> 3B gave monotonic gains, AUC beat SOTA, MCC nearly matched it) to
the MAIN pipeline for the first time. Round-3 scale-up already plateaued
for COACH420 and stabilized HOLO4K at parity -- but embedding capacity was
never tested here, unlike CryptoBench. Mirrors Phase52 exactly (same
dihedral logic, same round-3 topology blob) except ESM2 swapped to the
full 3B model (fp16, batch=1, matching Phase48's safety pattern -- main
pipeline sequences are capped at 1000 residues, even shorter than
CryptoBench's 1500 cap, so if anything this is safer)."""
import json, pickle
from pathlib import Path

C10 = pickle.load(open("/tmp/p10_cells.pkl", "rb"))
s10 = lambda i: (C10[i]["source"] if isinstance(C10[i]["source"], str) else "".join(C10[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 92 - Main Pipeline with ESM2-3B (the Untested Capacity Lever)

CryptoBench proved embedding capacity was a real, addressable lever (35M ->
650M -> 3B: AUC 0.867->0.882->0.887, MCC 0.362->0.376->0.416 at 8 seeds).
The main COACH420/HOLO4K pipeline has used ESM2-35M unchanged since
Phase 10 -- this is the one major improvement lever proven elsewhere that
was never applied here. Round-3 scale-up already plateaued (more DATA
stopped helping), so this tests whether more embedding CAPACITY picks up
where scale left off. Otherwise identical to Phase52 (same round-3
topology blob, same dihedral logic) -- only the ESM2 model changes.
''')

md(r'''## Cell 0 - Install dependencies''')
code(s10(2))

md(r'''## Cell 1 - Setup, load Phase 51's topology-fused pool''')
code(s10(3).replace('locate("phase6b_graphs.pt")', 'locate("phase51_graphs.pt")'))

md(r'''## Cell 2 - Recompute real backbone dihedrals (identical to Phase 10/30/37/52)''')
code(s10(5))

md(r'''## Cell 3 - ESM2-3B embeddings (fp16, batch=1 for memory safety -- matches Phase48's CryptoBench pattern; main-pipeline sequences capped at 1000 residues, shorter than CryptoBench's 1500 cap)''')
code(r'''
from transformers import AutoTokenizer, AutoModel

ESM_NAME = "facebook/esm2_t36_3B_UR50D"   # 3B params, hidden 2560 -- the
                                           # full model, matching CryptoBench's
                                           # own SOTA baseline and this
                                           # project's own best CryptoBench
                                           # result (Phase48/50/55).
tok = AutoTokenizer.from_pretrained(ESM_NAME)
esm = AutoModel.from_pretrained(ESM_NAME).half().to(DEVICE).eval()   # fp16: inference-only, no grads needed
ESM_DIM = esm.config.hidden_size
print(f"{ESM_NAME}: hidden {ESM_DIM}, device {DEVICE}, dtype {next(esm.parameters()).dtype}")

@torch.no_grad()
def embed_batch(seqs, max_len=1002):
    enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    out = esm(**enc).last_hidden_state          # (B, L+2, H) incl BOS/EOS
    embs = []
    for i, s in enumerate(seqs):
        n = len(s)
        embs.append(out[i, 1:1+n, :].float().cpu())   # strip BOS, keep exactly n residues
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
print(f"\nESM2 embeddings for {len(dataset):,} chains / {n_res:,} residues "
      f"in {(time.time()-t0)/60:.1f} min ({1000*(time.time()-t0)/n_res:.2f} ms/residue)")
del esm; torch.cuda.empty_cache()
''')

md(r'''## Cell 4 - Assemble augmented features, verify alignment, save (new output filename)''')
code(s10(9).replace('"/kaggle/working/phase10_graphs.pt"', '"/kaggle/working/phase92_graphs_esm3b.pt"')
           .replace('saved -> /kaggle/working/phase10_graphs.pt', 'saved -> /kaggle/working/phase92_graphs_esm3b.pt'))

md(r'''
---

`phase92_graphs_esm3b.pt` -- same schema as `phase52_graphs.pt` except
node dim is 2621 (21+6+2560+34) instead of 541. N_PRE/TOPO_DIM are read
dynamically from the blob, so `LocalizingEGNN`/`fit`/`run_epoch` apply
unchanged. Next: rerun the Phase20/53-style (COACH420) and
Phase24/54-style (HOLO4K) clean-training + benchmark pipelines on this
blob to see whether embedding capacity closes any remaining gap to P2Rank.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase92_esm_dihedral_esm3b.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
