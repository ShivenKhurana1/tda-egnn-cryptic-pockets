#!/usr/bin/env python3
"""Phase 288: scale the Phase 287 global-attention pilot up to the main pipeline. Pretrains the main
pipeline (35M scale, the original 2,056-chain pool, train/val/test splits and true_center already baked
into phase10_graphs.pt -- no extra data prep needed) with the SAME GlobalAttn block added after the EGCL
stack, 8 seeds, alongside a fresh from-scratch baseline rerun (not reusing old loc_deploy_seed checkpoints,
for a clean paired comparison). Produces checkpoints for Phase 289 to transfer onto CryptoBench, matching
how every other confirmed CryptoBench gain in this paper reached its final form (pretrain on the main
pipeline -> transfer -> compare against the existing headline).

Scale note: 35M, not 650M/3B, chosen deliberately given the weekly GPU quota is still exhausted (this runs
on CPU) and the main pipeline's pool (~1,700+ train chains after an internal val/test carve) is already
several times larger than CryptoBench's 700 -- 35M is the cheapest tier that still uses the real main-pipeline
training data and the real LocalizingEGNN architecture, matching this project's own standing discipline of
testing a new idea at the cheapest tractable scale before any further escalation.

usage: build_nb288.py <output_root>"""
import json, sys
from pathlib import Path

ROOT_OUT = Path(sys.argv[1]).resolve()
RESUME_FROM = int(sys.argv[2]) if len(sys.argv) > 2 else None   # resume: global_attn arm only, seeds RESUME_FROM..7 (12h kernel cap cut the first run at 2/8)
HERE = Path(__file__).resolve().parent
nb22 = json.load(open(HERE.parents[1] / "phase22_topology_localization_variance.ipynb"))
s22 = lambda i: "".join(nb22["cells"][i]["source"])

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md('''
# Phase 288 - Main-pipeline pretraining with global attention (35M, 8 seeds/arm)

Both arms (baseline h128L4, and baseline+global-attention) are trained fresh here on the real main-pipeline
data (phase10_graphs.pt: the original 2,056-chain pool, train/val/test splits and true_center already
present) so Phase 289 can transfer both sets of checkpoints onto CryptoBench for a clean, paired comparison
against the existing 35M-transfer headline.
''')
md("## Cell 1 — Install")
code("!pip install -q torch-geometric==2.6.1 biopython==1.84")
md("## Cell 2 — Setup: load phase10_graphs.pt (identical to Phase 22)")
code(s22(3))
md("## Cell 2b — Recover true ligand centers by fetching structures fresh from RCSB (identical to Phase 22; phase10_graphs.pt does NOT carry true_center -- this rebuilds it and also rebuilds splits with the 92/8 deployment-pretraining convention, matching Phase 266b)")
code(s22(4))
md("## Cell 3 — Model, with an optional global self-attention block after the EGCL stack (new parameters; from-scratch only)")
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
    """ONE multi-head self-attention block over every residue in a chain, regardless of 3D distance --
    unlike EGCL, which only passes messages along the radius graph's local edges. Operates entirely on the
    invariant scalar channel h (never reads or writes pos/coordinates): attention scores are dot-products of
    invariant features, the output is a weighted sum of invariant value vectors, so this block is trivially
    SE(3)-equivariant by construction -- still verified numerically below before any training. Identical to
    Phase 287's module (same pilot that motivated scaling this up)."""
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
            hg = h[idx].unsqueeze(0)                          # (1, n_g, h_dim)
            attn_out, _ = s.mha(hg, hg, hg)
            hg2 = s.norm1(hg + attn_out)
            hg3 = s.norm2(hg2 + s.ff(hg2))
            out[idx] = hg3.squeeze(0)
        return out

class LocalizingEGNN(nn.Module):
    """Same TopologicalEGNN as every prior phase, plus an optional global self-attention block after the
    EGCL stack (`use_global_attn=False` by default, reproducing the original architecture exactly when off)."""
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
md("## Cell 4 — Equivariance check (global-attn arm; rotate a batch, confirm the predicted center still rotates correspondingly)")
code(r'''
torch.manual_seed(0)
_probe = splits["val"][0].clone()
_m = LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=True, use_global_attn=True).to(DEVICE).eval()
_probe = _probe.to(DEVICE)
with torch.no_grad():
    _, c_ref = _m(_probe)
Q, _ = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(1)))
Q = (Q * torch.sign(torch.det(Q))).to(DEVICE)
shift = torch.tensor([17.3, -4.1, 62.0], device=DEVICE)
_rot = _probe.clone(); _rot.pos = _probe.pos @ Q.T + shift
with torch.no_grad():
    _, c_rot = _m(_rot)
c_ref_transformed = c_ref @ Q.T + shift
err = (c_rot - c_ref_transformed).norm().item()
print(f"equivariance check (global attention): max error {err:.2e} (expect ~1e-5, matching every other equivariance check in this project)")
assert err < 1e-3, "equivariance broken -- do not proceed to training"
del _m, _probe, _rot, c_ref, c_rot
''')
md("## Cell 5 — Training machinery (Phase 22, unchanged, use_global_attn threaded through fit())")
code(s22(8).replace("model = LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=use_loc_head).to(DEVICE)",
                    "model = LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=use_loc_head, use_global_attn=use_global_attn).to(DEVICE)")
          .replace("def fit(use_loc_head, lam, seed, epochs=40, patience=8, batch=16):",
                   "def fit(use_loc_head, lam, seed, use_global_attn=False, epochs=40, patience=8, batch=16):"))
md("## Cell 6 — Train BOTH arms (baseline rerun + global-attn) x 8 seeds from scratch; save checkpoints")
code(r'''
from pathlib import Path as _P
WORK = _P("/kaggle/working"); (WORK/"cache").mkdir(parents=True, exist_ok=True)

BEST_LAMBDA = 0.1
ARMS = ARMS_PLACEHOLDER
SEEDS = SEEDS_PLACEHOLDER
for arm_name, use_ga in ARMS:
    log = []; t0 = time.time()
    for seed in SEEDS:
        model, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed, use_global_attn=use_ga)
        v = run_epoch(model, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
        torch.save(model.state_dict(), WORK / f"cache/pretrain288_{arm_name}_seed{seed}.pt")
        log.append({"arm": arm_name, "seed": seed, "val_pr": float(val_pr), "val_dcc": float(v.get("dcc", float("nan")))})
        json.dump(log, open(WORK / f"phase288_{arm_name}_train_log.json", "w"), indent=2)
        print(f"  [{arm_name}] seed {seed}: val PR {val_pr:.4f}  val DCC {v.get('dcc', float('nan')):.4f}  [{(time.time()-t0)/60:.1f}m]", flush=True)
    print(f"{arm_name} done\n")
print("all arms done; checkpoints in cache/")
'''.replace('ARMS_PLACEHOLDER', repr([('global_attn', True)] if RESUME_FROM is not None else [('base', False), ('global_attn', True)])).replace('SEEDS_PLACEHOLDER', repr(list(range(RESUME_FROM, 8)) if RESUME_FROM is not None else list(range(8)))))

nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
SLUG = "288r" if RESUME_FROM is not None else "288"
d = ROOT_OUT / SLUG; d.mkdir(parents=True, exist_ok=True)
o = d / "phase288_pretrain_35m_global_attention.ipynb"; o.write_text(json.dumps(nb, indent=1) + "\n")
meta = {"id": "blackpearls1/phase288" + ("r-resume-global-attention" if RESUME_FROM is not None else "-pretrain-35m-global-attention"), "title": "Phase288" + ("r Resume Global Attention" if RESUME_FROM is not None else " Pretrain 35m Global Attention"), "code_file": o.name,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": False, "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["blackpearls1/tda-egnn-phase10-esm-boost"]}
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote", o)
