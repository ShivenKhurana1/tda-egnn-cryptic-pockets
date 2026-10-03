#!/usr/bin/env python3
"""Phase 289: transfer Phase 288's main-pipeline-pretrained (35M, with/without global attention) checkpoints
onto CryptoBench, completing the scale-up the Phase 287 pilot asked for. Both arms (baseline-transfer and
global-attn-transfer) are fine-tuned here, 8 seeds each, each seed initialised from its OWN Phase 288 seed-
matched checkpoint (seed s <- pretrained seed s, matching every other transfer-learning phase in this
project). Comparison target: the existing 35M-transfer headline (Phase 149/122 lineage, cached) is NOT
rerun -- this produces a fresh baseline-transfer arm instead, trained in the exact same kernel as the new
global-attn-transfer arm, for a clean paired comparison that removes any cross-kernel inconsistency risk
(same discipline as Phase 284/286/287).

usage: build_nb289.py <output_root>"""
import json, sys
from pathlib import Path

ROOT_OUT = Path(sys.argv[1]).resolve()
HERE = Path(__file__).resolve().parent
nb149 = json.load(open(HERE.parents[1] / "phase149_cryptobench_35m_transfer_eval.ipynb"))
s149 = lambda i: "".join(nb149["cells"][i]["source"])
nb129 = json.load(open(HERE.parents[1] / "phase129_cryptobench_esm3b_transfer.ipynb"))
s129 = lambda i: "".join(nb129["cells"][i]["source"])

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md('''
# Phase 289 - CryptoBench transfer from Phase 288's global-attention main-pipeline checkpoints (35M, 8 seeds/arm)

Completes the scale-up: Phase 287 found a promising (partially confirmed) signal for global attention tested
from scratch directly on CryptoBench; Phase 288 pretrained the main pipeline itself (35M) with the same
change; this transfers both sets of checkpoints onto CryptoBench, exactly the path every other confirmed
CryptoBench gain in this paper took. Both arms (baseline-transfer, global-attn-transfer) trained fresh here.
''')
md("## Cell 1 — Install")
code("!pip install -q torch-geometric==2.6.1")
md("## Cell 2 — Setup: load the 35M CryptoBench blob (identical to Phase 149)")
code(s149(4))
md("## Cell 3 — Model, with an optional global self-attention block after the EGCL stack (identical to Phase 288)")
code(r'''
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

class GlobalAttn(nn.Module):
    """Identical to Phase 287/288's module: one global multi-head self-attention block operating only on
    the invariant scalar channel h (never reads/writes pos), trivially SE(3)-equivariant by construction."""
    def __init__(s, h, n_heads=4):
        super().__init__()
        s.mha = nn.MultiheadAttention(h, n_heads, batch_first=True)
        s.norm1 = nn.LayerNorm(h)
        s.ff = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
        s.norm2 = nn.LayerNorm(h)

    def forward(s, h, batch):
        n_graphs = int(batch.max().item()) + 1
        out = h.clone()
        for g in range(n_graphs):
            idx = (batch == g).nonzero(as_tuple=True)[0]
            hg = h[idx].unsqueeze(0)
            attn_out, _ = s.mha(hg, hg, hg)
            hg2 = s.norm1(hg + attn_out)
            hg3 = s.norm2(hg2 + s.ff(hg2))
            out[idx] = hg3.squeeze(0)
        return out

class LocalizingEGNN(nn.Module):
    """Same TopologicalEGNN as every prior phase, plus an optional global self-attention block after the
    EGCL stack (`use_global_attn=False` by default, reproducing the original architecture exactly when off).
    Identical to Phase 287/288's class; checkpoints from either phase load into this unchanged."""
    def __init__(s, n_pre, td, h=128, L=4, dr=0.1, use_loc_head=True, use_global_attn=False):
        super().__init__(); s.n_pre=n_pre; s.use_loc_head=use_loc_head; s.use_global_attn=use_global_attn
        s.embed=nn.Sequential(nn.Linear(n_pre+td,h),nn.SiLU(),nn.Linear(h,h))
        s.ls=nn.ModuleList([EGCL(h,uc=(i<L-1)) for i in range(L)])
        s.fs=nn.ModuleList([Fus(h,td) for _ in range(L)])
        if use_global_attn: s.global_attn = GlobalAttn(h)
        s.head=nn.Sequential(nn.Linear(h,h//2),nn.SiLU(),nn.Dropout(dr),nn.Linear(h//2,1))
        if use_loc_head:
            s.loc_head=nn.Sequential(nn.Linear(h,h//2),nn.SiLU(),nn.Linear(h//2,1))

    def forward(s, d):
        h=s.embed(d.x); pos=d.pos; t=d.x[:,s.n_pre:]
        for l,f in zip(s.ls,s.fs): h,pos=l(h,pos,d.edge_index,d.edge_attr); h=f(h,t)
        batch = d.batch if hasattr(d, "batch") and d.batch is not None \
                else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        if s.use_global_attn: h = s.global_attn(h, batch)
        logits = s.head(h).squeeze(-1)
        pred_center = None
        if s.use_loc_head:
            loc_logit = s.loc_head(h).squeeze(-1)
            attn = pyg_softmax(loc_logit, batch)                    # sums to 1 per graph
            weighted = attn.unsqueeze(-1) * pos                     # (N,3)
            n_graphs = int(batch.max().item()) + 1
            pred_center = torch.zeros(n_graphs, 3, device=h.device).index_add_(0, batch, weighted)
        return logits, pred_center
''')
md("## Cell 4 — Training machinery: fit() optionally loads a pretrained state_dict, use_global_attn threaded through (Phase 129, adapted)")
code(s129(8).replace(
    "def fit(use_loc_head, lam, seed, pretrained_path=None, epochs=40, patience=8, batch=16):\n    torch.manual_seed(seed); np.random.seed(seed)\n    model = LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=use_loc_head).to(DEVICE)",
    "def fit(use_loc_head, lam, seed, pretrained_path=None, use_global_attn=False, epochs=40, patience=8, batch=16):\n    torch.manual_seed(seed); np.random.seed(seed)\n    model = LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=use_loc_head, use_global_attn=use_global_attn).to(DEVICE)"
))
md("## Cell 5 — Transfer-finetune BOTH arms x 8 seeds from Phase 288's seed-matched checkpoints; save per-seed val+test predictions")
code(r'''
@torch.no_grad()
def predict(model, split):
    out = {}
    for d in splits[split]:
        b = d.clone().to(DEVICE); logits, pc = model(b)
        out[(str(d.pdb_id).lower(), str(d.chain_id))] = dict(p=torch.sigmoid(logits).cpu().numpy().astype(np.float32),
                                                             c=pc.cpu().numpy()[0], y=d.y.numpy().astype(np.int8),
                                                             tc=d.true_center.numpy())
    return out

BEST_LAMBDA = 0.1
for arm_name, use_ga in [("base", False), ("global_attn", True)]:
    PREDS = {"arm": arm_name, "seeds": {}}; t0 = time.time()
    for seed in range(8):
        ck = locate(f"pretrain288_{arm_name}_seed{seed}.pt")
        model, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed, pretrained_path=ck, use_global_attn=use_ga)
        PREDS["seeds"][seed] = {"val": predict(model, "val"), "test": predict(model, "test"), "val_pr": float(val_pr)}
        torch.save(PREDS, WORK / f"phase289_{arm_name}_preds.pt")   # incremental: a late crash loses one seed, not all
        print(f"  [{arm_name}] seed {seed}: val PR {val_pr:.4f}  [{(time.time()-t0)/60:.1f}m]", flush=True)
    print(f"{arm_name} done\n")
print("all arms done")
''')

nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
d = ROOT_OUT / "289"; d.mkdir(parents=True, exist_ok=True)
o = d / "phase289_cb35m_global_attention_transfer.ipynb"; o.write_text(json.dumps(nb, indent=1) + "\n")
meta = {"id": "blackpearls1/phase289-cb35m-global-attention-transfer", "title": "Phase289 Cb35m Global Attention Transfer", "code_file": o.name,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": False, "enable_tpu": False, "enable_internet": True, "dataset_sources": ["blackpearls1/phase288-checkpoints-35m"], "competition_sources": [],
        # the cancelled Phase 288 kernel is rejected as a kernel source, so all 16 checkpoints (288 base 0-7, global-attn 0-1; 288r global-attn 2-7) ship as one dataset
        "kernel_sources": ["blackpearls1/tda-egnn-phase42-cryptobench-ingestion"]}
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote", o)
