#!/usr/bin/env python3
"""Phase 287: a global (long-range) attention mechanism in the trunk, tested specifically on CryptoBench,
from scratch -- the genuinely new, untried architectural lever (as opposed to GNN depth, which was already
tried at this exact scale, Phase 154, and honestly reported as borderline/not-established; re-running it now
with more seeds to try to force significance would be the p-hacking this project explicitly declined to do,
so it is NOT repeated here).

Motivation: every layer in this architecture (EGCL) is local message passing over an 8-10A radius graph. The
GNN-capacity sweep already showed that more WIDTH/DEPTH within this same local-receptive-field paradigm does
not help (main pipeline: clean null; CryptoBench: the one ambiguous positive, not re-chased here). That rules
out capacity, but not RECEPTIVE FIELD: a residue can currently only be informed by a distant part of the fold
through many diluted hops. A real attention mechanism that lets every residue attend to every other residue
directly, regardless of distance, is a qualitatively different lever, untested anywhere in this project, and
could plausibly capture long-range allosteric coupling relevant to pocket-opening dynamics that local message
passing structurally cannot represent well even with unlimited depth.

Design: ONE global multi-head self-attention block, inserted after the full EGCL+Fus stack (so good local
features are already formed), operating ENTIRELY on the invariant scalar channel h -- it never reads or
writes `pos`. This makes it trivially SE(3)-equivariant by construction (attention scores are dot-products of
invariant features; the output is a weighted combination of invariant value vectors; no coordinate information
enters or leaves the block), verified numerically before training anyway, matching this project's standard
practice. This necessarily adds new parameters, so -- like the topological-variability and ANM-flexibility
experiments, and unlike top-K/local-attention which reused existing pretrained checkpoints -- it must be
trained from scratch; CryptoBench itself (not the much larger, much more expensive main pipeline) is the
right place to test it first, exactly how GNN depth was tested, for a cheap, clean, isolated comparison.

Both arms (baseline h128L4, and h128L4+global-attn) are trained FRESH in this one kernel, 8 seeds each, saving
full per-seed val+test predictions and centers (not just pooled-ensemble point estimates, unlike Phase 154) --
rerunning the baseline here instead of reusing Phase 154/46/47's older point-estimate-only numbers makes a
properly powered, two-test-rule paired comparison possible, and removes any cross-run inconsistency risk.

usage: build_nb287.py <output_root>"""
import json, sys
from pathlib import Path

ROOT_OUT = Path(sys.argv[1]).resolve()
HERE = Path(__file__).resolve().parent
nb154 = json.load(open(HERE.parents[1] / "phase154_cryptobench_depth_sweep.ipynb"))
s154 = lambda i: "".join(nb154["cells"][i]["source"])

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md('''
# Phase 287 - Global (long-range) attention in the trunk, tested on CryptoBench from scratch (650M, 8 seeds/arm)

Baseline (h128L4, no global attention) is RERUN fresh here (not reused from Phase 154/46/47) so both arms get
full per-seed val+test predictions in the same format, enabling a proper paired two-test-rule comparison.
Everything else -- data, loss, optimiser, early stopping -- is identical to Phase 154.
''')
md("## Cell 1 - Install")
code("!pip install -q torch-geometric==2.6.1")
md("## Cell 2 - Setup: load Phase 45's CryptoBench-650M blob (identical to Phase 154)")
code(s154(4))
md("## Cell 3 - Model, with an optional global self-attention block after the EGCL stack (new parameters; from-scratch only)")
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
    SE(3)-equivariant by construction -- still verified numerically below before any training."""
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
md("## Cell 4 - Equivariance check (global-attn arm; rotate a batch, confirm the predicted center still rotates correspondingly)")
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
md("## Cell 5 - Training machinery (Phase 154, unchanged)")
code(s154(8).replace("model = LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=use_loc_head, L=L).to(DEVICE)",
                     "model = LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=use_loc_head, L=L, use_global_attn=use_global_attn).to(DEVICE)")
          .replace("def fit(use_loc_head, lam, seed, L=4, epochs=40, patience=8, batch=16):",
                   "def fit(use_loc_head, lam, seed, L=4, use_global_attn=False, epochs=40, patience=8, batch=16):"))
md("## Cell 6 - Train BOTH arms (baseline rerun + global-attn) x 8 seeds from scratch; save per-seed val+test predictions and centers")
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
        model, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed, L=4, use_global_attn=use_ga)
        PREDS["seeds"][seed] = {"val": predict(model, "val"), "test": predict(model, "test"), "val_pr": float(val_pr)}
        torch.save(PREDS, WORK / f"phase287_{arm_name}_preds.pt")   # incremental: a late crash loses one seed, not all
        print(f"  [{arm_name}] seed {seed}: val PR {val_pr:.4f}  [{(time.time()-t0)/60:.1f}m]", flush=True)
    print(f"{arm_name} done\n")
print("all arms done")
''')

nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
d = ROOT_OUT / "287"; d.mkdir(parents=True, exist_ok=True)
o = d / "phase287_cb650m_global_attention.ipynb"; o.write_text(json.dumps(nb, indent=1) + "\n")
meta = {"id": "blackpearls1/phase287-cb650m-global-attention", "title": "Phase287 Cb650m Global Attention", "code_file": o.name,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        # CPU: weekly GPU quota remains exhausted (Phase 285/286). The 650M blob is already ESM-embedded, so
        # this kernel only trains the EGNN itself from scratch -- feasible on CPU, just slower.
        "enable_gpu": False, "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["blackpearls1/tda-egnn-phase45-cryptobench-esm650m"]}
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote", o)
