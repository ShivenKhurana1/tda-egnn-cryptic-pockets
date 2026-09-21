#!/usr/bin/env python3
"""Phase 162: score-only TEST eval of Phase161's top-K localization
checkpoints, matching Phase150's exact protocol for a direct comparison
against the existing 3B-transfer baseline (test AUC 0.9111, MCC 0.4664,
DCC 0.3077)."""
import json
from pathlib import Path

nb150 = json.load(open("/Users/shivenkhurana/bioproj/phase150_cryptobench_3b_transfer_eval.ipynb"))
c150 = nb150["cells"]
s150 = lambda i: (c150[i]["source"] if isinstance(c150[i]["source"], str) else "".join(c150[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 162 — CryptoBench Top-K Localization: Score-Only TEST Eval

Loads the 8 Phase161 top-K (K=20) localization checkpoints, evaluates
on TEST with a val-tuned threshold -- matches Phase150's exact protocol
for a direct comparison against the existing 3B-transfer baseline (test
AUC 0.9111, MCC 0.4664, DCC 0.3077).
''')

md(r'''## Cell 1 — Install dependencies''')
code(s150(2))

md(r'''## Cell 2 — Setup, load the 3B CryptoBench blob''')
code(s150(4))

md(r'''## Cell 3 — Model, with the top-K localization readout (identical parameters, only the readout logic differs from every prior phase)''')
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

class LocalizingEGNN(nn.Module):
    def __init__(s, n_pre, td, h=128, L=4, dr=0.1, use_loc_head=True, topk=None):
        super().__init__(); s.n_pre=n_pre; s.use_loc_head=use_loc_head; s.topk=topk
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
            if s.topk is not None:
                n_graphs_ = int(batch.max().item()) + 1
                masked = loc_logit.clone()
                for g in range(n_graphs_):
                    idx = (batch == g).nonzero(as_tuple=True)[0]
                    if len(idx) > s.topk:
                        vals = loc_logit[idx]
                        _, top_local = torch.topk(vals, s.topk)
                        keep_mask = torch.zeros(len(idx), dtype=torch.bool, device=h.device)
                        keep_mask[top_local] = True
                        masked[idx[~keep_mask]] = float("-inf")
                loc_logit = masked
            attn = pyg_softmax(loc_logit, batch)
            weighted = attn.unsqueeze(-1) * pos
            n_graphs = int(batch.max().item()) + 1
            pred_center = torch.zeros(n_graphs, 3, device=h.device).index_add_(0, batch, weighted)
        return logits, pred_center
''')

md(r'''## Cell 4 — Load the 8 top-K (K=20) checkpoints into an ensemble''')
code(r'''
ensemble = []
for seed in range(8):
    m = LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=True, topk=20).to(DEVICE)
    state = torch.load(locate(f"cryptobench_esm3b_topk_seed{seed}.pt"), map_location=DEVICE)
    m.load_state_dict(state)
    m.eval()
    ensemble.append(m)
print(f"loaded {len(ensemble)} top-K (K=20) checkpoints into ensemble")
''')

md(r'''## Cell 5 — Pick a decision threshold on VAL ONLY (never touch test for this), then evaluate on TEST''')
code(s150(10))

md(r'''## Cell 6 — Field-standard residue classification metrics on TEST (pooled)''')
code(s150(12))

md(r'''## Cell 7 — Our own localization-head metric''')
code(s150(14).replace('phase150_summary.json', 'phase162_topk_eval_summary.json')
             .replace('saved -> phase150_summary.json', 'saved -> phase162_topk_eval_summary.json'))

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase162_cryptobench_topk_eval.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
