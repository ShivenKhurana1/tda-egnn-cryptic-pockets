#!/usr/bin/env python3
"""Phase 123: CryptoBench cross-scale ensemble. The 3 already-trained
ESM scales (35M: Phase42-44, AUC 0.8717; 650M: Phase45-47, AUC 0.8873; 3B:
Phase48/50/55, AUC 0.8878) were each independently 8-seed-ensembled
WITHIN their own scale. Never combined ACROSS scales. Different embedding
sizes may make different, partially-uncorrelated errors -- averaging
their predictions could beat any single scale, at ZERO new training cost
(pure inference on already-trained checkpoints + already-computed
feature blobs, no live ESM needed). Verifies (pdb_id, chain_id) test-set
alignment across the 3 independently-ingested blobs before combining
(joblib preserves input order, but this is confirmed, not assumed)."""
import json
from pathlib import Path

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 123 — CryptoBench: Cross-Scale Ensemble (35M + 650M + 3B)

Each ESM scale was independently 8-seed-ensembled: 35M AUC 0.8717, 650M
0.8873, 3B 0.8878 (monotonic but diminishing gains). Never combined
ACROSS scales. Pure inference on already-trained checkpoints -- zero new
training. If different scales make different errors, this could beat
3B's 0.8878 for free.
''')

md(r'''## Cell 1 — Install dependencies''')
code(r'''!pip install -q torch-geometric==2.6.1''')

md(r'''## Cell 2 — Model (identical to every CryptoBench phase)''')
code(r'''
import json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax as pyg_softmax
from sklearn.metrics import (average_precision_score, roc_auc_score,
                              matthews_corrcoef, f1_score)

def locate(f):
    c=[]
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): c.extend(r.rglob(f))
    if not c: raise FileNotFoundError(f)
    return max(c, key=lambda p: p.stat().st_size)

GPU_OK=False
if torch.cuda.is_available():
    try:
        _p=torch.randn(256,256,device="cuda"); (_p@_p).sum().item(); torch.cuda.synchronize(); GPU_OK=True
        print(f"GPU: {torch.cuda.get_device_name(0)} (probe PASS)")
    except RuntimeError as e: print("GPU unusable:", e)
DEVICE = torch.device("cuda" if GPU_OK else "cpu")
WORK = Path("/kaggle/working"); (WORK/"cache").mkdir(parents=True, exist_ok=True)

class EGCL(nn.Module):
    def __init__(s,h,ea=2,act=nn.SiLU,uc=True):
        super().__init__(); s.uc=uc
        s.e=nn.Sequential(nn.Linear(2*h+1+ea,h),act(),nn.Linear(h,h),act())
        s.n=nn.Sequential(nn.Linear(2*h,h),act(),nn.Linear(h,h))
        s.a=nn.Sequential(nn.Linear(h,1),nn.Sigmoid())
        if uc:
            s.c=nn.Sequential(nn.Linear(h,h),act(),nn.Linear(h,1,bias=False))
            nn.init.xavier_uniform_(s.c[-1].weight,gain=0.001)
        s.norm=nn.LayerNorm(h)
    def forward(s,h,x,ei,ea):
        src,dst=ei; rel=x[src]-x[dst]; d2=(rel**2).sum(-1,keepdim=True)
        m=s.e(torch.cat([h[src],h[dst],d2,ea],-1)); m=m*s.a(m)
        agg=torch.zeros_like(h).index_add_(0,dst,m)
        deg=torch.zeros(h.size(0),1,device=h.device).index_add_(0,dst,torch.ones_like(d2)).clamp(min=1)
        h=s.norm(h+s.n(torch.cat([h,agg/deg],-1)))
        if s.uc:
            w=s.c(m)*rel/(d2.sqrt()+1.0); x=x+torch.zeros_like(x).index_add_(0,dst,w)/deg
        return h,x

class Fus(nn.Module):
    def __init__(s,h,t):
        super().__init__(); s.p=nn.Linear(t,h); s.g=nn.Sequential(nn.Linear(h+t,h),nn.Sigmoid())
    def forward(s,h,t): return h+s.g(torch.cat([h,t],-1))*s.p(t)

class LocalizingEGNN(nn.Module):
    def __init__(s, n_pre, td, h=128, L=4, dr=0.1, use_loc_head=True):
        super().__init__(); s.n_pre=n_pre; s.use_loc_head=use_loc_head
        s.embed=nn.Sequential(nn.Linear(n_pre+td,h),nn.SiLU(),nn.Linear(h,h))
        s.ls=nn.ModuleList([EGCL(h,uc=(i<L-1)) for i in range(L)])
        s.fs=nn.ModuleList([Fus(h,td) for _ in range(L)])
        s.head=nn.Sequential(nn.Linear(h,h//2),nn.SiLU(),nn.Dropout(dr),nn.Linear(h//2,1))
        if use_loc_head:
            s.loc_head=nn.Sequential(nn.Linear(h,h//2),nn.SiLU(),nn.Linear(h//2,1))
    def forward(s, d):
        h=s.embed(d.x); pos=d.pos; t=d.x[:,s.n_pre:]
        for l,f in zip(s.ls,s.fs): h,pos=l(h,pos,d.edge_index,d.edge_attr); h=f(h,t)
        logits = s.head(h).squeeze(-1)
        pred_center = None
        if s.use_loc_head:
            batch = d.batch if hasattr(d, "batch") and d.batch is not None \
                    else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            loc_logit = s.loc_head(h).squeeze(-1)
            attn = pyg_softmax(loc_logit, batch)
            weighted = attn.unsqueeze(-1) * pos
            n_graphs = int(batch.max().item()) + 1
            pred_center = torch.zeros(n_graphs, 3, device=h.device).index_add_(0, batch, weighted)
        return logits, pred_center

@torch.no_grad()
def predict_seed(model, loader):
    """Returns pooled (probs, y) in loader order. Alignment across scales
    is verified separately, per-CHAIN (not per-residue) directly from the
    unbatched dataset list -- PyG's DataLoader collates non-tensor
    attributes like pdb_id/chain_id per-GRAPH, not per-NODE, so zipping
    them against per-residue probs inside the batched loop would silently
    misalign (length mismatch: one entry per chain vs one per residue)."""
    all_probs, all_y = [], []
    for b in loader:
        b = b.to(DEVICE)
        lg, pc = model(b)
        all_probs.append(torch.sigmoid(lg).cpu().numpy())
        all_y.append(b.y.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_y)
''')

md(r'''## Cell 3 — Load each scale's blob + all 8 checkpoints, predict on test (pure inference, no training)''')
code(r'''
SCALES = {
    "35M":  dict(blob="phase42_cryptobench_graphs.pt",       ckpt_prefix="cryptobench_deploy_seed"),
    "650M": dict(blob="phase45_cryptobench_graphs_esm650m.pt", ckpt_prefix="cryptobench_esm650m_deploy_seed"),
    "3B":   dict(blob="phase48_cryptobench_graphs_esm3b.pt",   ckpt_prefix="cryptobench_esm3b_deploy_seed"),
}

scale_test_probs = {}   # scale -> 8-seed-averaged test probs (aligned by key)
scale_test_keys = {}
scale_test_y = {}
t0 = time.time()
for scale, cfg in SCALES.items():
    blob = torch.load(locate(cfg["blob"]), map_location="cpu", weights_only=False)
    dataset = blob["dataset"]
    N_PRE = int(blob["n_pre_topo"]); TOPO_DIM = int(blob["topo_dim"])
    test_set = [d for d in dataset if d.split == "test"]
    test_loader = DataLoader(test_set, batch_size=16)
    # per-chain alignment key, built directly from the unbatched list (not
    # through the loader -- see predict_seed's docstring)
    scale_test_keys[scale] = [(d.pdb_id, d.chain_id, int(d.num_nodes)) for d in test_set]

    seed_probs = []
    for seed in range(8):
        ckpt = locate(f"{cfg['ckpt_prefix']}{seed}.pt")
        m = LocalizingEGNN(N_PRE, TOPO_DIM).to(DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE)); m.eval()
        p, y = predict_seed(m, test_loader)
        seed_probs.append(p)
        if seed == 0:
            scale_test_y[scale] = y
    scale_test_probs[scale] = np.mean(seed_probs, axis=0)
    print(f"  {scale:<6} 8-seed test predictions ready ({len(test_set)} chains, "
          f"{len(scale_test_y[scale])} residues)  [{(time.time()-t0)/60:.1f}m]")

# verify (pdb_id, chain_id) alignment across scales before combining positionally
ref_keys = scale_test_keys["35M"]
for scale in ("650M", "3B"):
    assert scale_test_keys[scale] == ref_keys, \
        f"{scale} test-set key order does NOT match 35M -- would need to realign by key, not just concatenate"
print(f"\nalignment verified: all 3 scales' test sets have identical (pdb_id,chain_id) order, "
      f"{len(ref_keys)} residues")
''')

md(r'''## Cell 4 — Cross-scale ensemble (average the 3 scales'' own 8-seed-ensembled probabilities), compare to each scale alone''')
code(r'''
def metrics_at(p, y, thresh):
    pred = (p >= thresh).astype(int)
    return {"auc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
            "mcc": matthews_corrcoef(y, pred), "f1": f1_score(y, pred)}

def best_thresh(p, y):
    best_t, best_mcc = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        mcc = matthews_corrcoef(y, (p >= t).astype(int))
        if mcc > best_mcc: best_mcc, best_t = mcc, t
    return best_t

y_ref = scale_test_y["35M"]   # identical across scales, verified above

results = {}
for scale in ("35M", "650M", "3B"):
    p = scale_test_probs[scale]
    t = best_thresh(p, y_ref)   # NOTE: tuned on TEST here for a quick same-run comparison;
                                 # Phase44/47/55's own numbers (val-tuned) are the ones to cite for real claims
    results[scale] = metrics_at(p, y_ref, t)

ens_35_650 = (scale_test_probs["35M"] + scale_test_probs["650M"]) / 2
ens_650_3b = (scale_test_probs["650M"] + scale_test_probs["3B"]) / 2
ens_all3 = (scale_test_probs["35M"] + scale_test_probs["650M"] + scale_test_probs["3B"]) / 3
for name, p in [("35M+650M", ens_35_650), ("650M+3B", ens_650_3b), ("35M+650M+3B", ens_all3)]:
    t = best_thresh(p, y_ref)
    results[name] = metrics_at(p, y_ref, t)

print(f"{'variant':<16}{'AUC':>8}{'AUPRC':>8}{'MCC':>8}{'F1':>8}")
print("-"*48)
for name, m in results.items():
    print(f"{name:<16}{m['auc']:>8.4f}{m['auprc']:>8.4f}{m['mcc']:>8.4f}{m['f1']:>8.4f}")

print(f"\nfor reference, each scale's OWN val-tuned 8-seed number (Phase44/47/55, the citable ones):")
print(f"  35M:  AUC 0.8717  MCC 0.3674")
print(f"  650M: AUC 0.8873  MCC 0.3986")
print(f"  3B:   AUC 0.8878  MCC 0.4157")

json.dump(results, open(WORK/"phase123_cross_scale_ensemble.json", "w"), indent=2)
print("\nsaved -> phase123_cross_scale_ensemble.json")
''')

md(r'''
---

## Reading this

- **35M+650M+3B (or any combo) clearly beats 3B alone** (AUC/MCC
  outside typical seed noise) -- different scales make complementary
  errors, a genuine free improvement worth adopting as the final
  CryptoBench number (re-run with a proper val-tuned threshold for the
  citable claim).
- **Ties 3B alone** -- the scales are too correlated (all trained on the
  same structures/labels, just different embeddings) to gain from
  combining.
- **Worse than 3B alone** -- the weaker scales (especially 35M) are
  dragging the ensemble down; try 650M+3B only, or weight by scale
  quality instead of equal averaging.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase123_cryptobench_cross_scale_ensemble.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
