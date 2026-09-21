#!/usr/bin/env python3
"""Phase 14: auxiliary localization loss - validate cheaply on our own held-out
split before spending another 3 hours re-running the external SOTA benchmark."""
import json
from pathlib import Path

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 14 — Training the Model to Localize, Not Just Classify

Phase 13 showed heuristic tuning on the residue-to-center conversion recovers
real accuracy (DCC 0.377 -> 0.435 on COACH420/HOLO4K) but doesn't close the gap
to P2Rank (still -0.075). The likely reason: nothing in training ever told the
model "these residues, once weighted together, should centroid near the exact
ligand position" — it only ever learned "is this residue near a ligand,
yes/no". This notebook adds that missing signal directly.

**Deliberately scoped cheap first.** Re-validating on COACH420/HOLO4K again
costs ~3 hours per attempt (that cost is in the topology computation, not
anything specific to this idea). Before paying that again, this notebook tests
the auxiliary-loss idea on our **own held-out test split** (the same
fixed 70/15/15 split used since Phase 7/8/10, already fully featurized in
`phase10_graphs.pt` — no re-featurization needed). If it doesn't help here, it
won't help on the harder external benchmark either, and there is no reason to
spend the 3 hours finding that out the expensive way.

### The architecture change

A second small head produces a per-residue "localization logit", pooled with a
**batch-aware softmax** (`torch_geometric.utils.softmax`, grouped by graph — not
an independent per-residue sigmoid like the classification head) into weights
that sum to 1 within each chain. The predicted pocket center is the resulting
weighted average of CA positions:

$$c_{\text{pred}} = \sum_i \text{softmax}(\ell_i) \cdot x_i$$

This is trained against the true ligand center of mass with a Huber loss
(robust to the occasional mismatched/ambiguous ligand), added to the existing
per-residue BCE loss.

**Why a separate head, not reusing the classification probability itself:**
BCE wants a *calibrated contact probability* (is this residue within 4.5 Å of
any part of the ligand); the centroid task wants a *sharply peaked localization
weight* concentrated near the ligand's exact center. A residue touching the far
edge of a large ligand should score high on the first and low on the second.
Forcing one scalar to serve both creates conflicting gradients. Two small heads
sharing the same backbone avoids that.

**Why this stays equivariant:** a weighted average of equivariant coordinates
$x_i$ with rotation/translation-*invariant* weights (softmax over invariant
per-residue logits) is itself equivariant — the same principle behind every
coordinate operation in this architecture since Phase 3. Verified numerically
below, not just asserted.
''')

md(r'''## Cell 1 — Setup''')

code(r'''
!pip install -q torch-geometric==2.6.1 biopython==1.84
''')

code(r'''
import gzip, json, os, time, urllib.request, warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax as pyg_softmax
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

warnings.filterwarnings("ignore")

def locate(f):
    c=[]
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): c.extend(r.rglob(f))
    if not c: raise FileNotFoundError(f)
    return max(c, key=lambda p: p.stat().st_size)

blob = torch.load(locate("phase10_graphs.pt"), map_location="cpu", weights_only=False)
dataset = blob["dataset"]
N_PRE = int(blob["n_pre_topo"]); TOPO_DIM = int(blob["topo_dim"]); POS_WEIGHT = float(blob["pos_weight"])

GPU_OK=False
if torch.cuda.is_available():
    try:
        _p=torch.randn(256,256,device="cuda"); (_p@_p).sum().item(); torch.cuda.synchronize(); GPU_OK=True
        print(f"GPU: {torch.cuda.get_device_name(0)} (probe PASS)")
    except RuntimeError as e: print("GPU unusable:", e)
DEVICE = torch.device("cuda" if GPU_OK else "cpu")

splits = {s: [d for d in dataset if d.split == s] for s in ("train","val","test")}
print(f"device {DEVICE} | node dim {dataset[0].x.size(1)} | " +
      " | ".join(f"{k} {len(v)}" for k,v in splits.items()))
''')

md(r'''
## Cell 2 — Recover true ligand centers (never stored on the graphs)

`d.pos` is CA coordinates already centered to zero mean at graph-build time, but
the centering offset itself was never saved. Re-parsing recovers both the raw
(uncentered) CA positions and the raw ligand atoms in the same frame, so the
offset can be computed and applied to the ligand — landing it in exactly the
frame `d.pos` already uses. The geometry-drift check (same pattern as every
other phase that re-parses these structures) catches any residue-order
mismatch before it can silently corrupt a training target.
''')

code(r'''
from Bio.PDB import PDBParser

AA3_TO_1={"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
          "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
          "THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M"}
NON_LIGAND={"HOH","DOD","WAT","SO4","PO4","GOL","EDO","PEG","PG4","PGE","1PE","MPD","TRS",
            "ACT","ACY","FMT","CIT","MES","EPE","DMS","IMD","BME","NO3","AZI","CO3","IOD",
            "BR","CL","F","NA","K","CS","RB","LI","UNX","UNL","UNK"}
CATALYTIC_METALS={"ZN","MG","MN","FE","FE2","CU","CU1","CA","NI","CO","CD","W","MO"}
MIN_LIG_ATOMS = 6
RAW = Path("/tmp/pdb_raw"); RAW.mkdir(parents=True, exist_ok=True)

need = sorted({d.pdb_id for d in dataset})
def fetch(pid):
    for ext in ("pdb","cif"):
        p = RAW / f"{pid}.{ext}.gz"
        if p.exists() and p.stat().st_size > 500: return pid, p
    for ext in ("pdb","cif"):
        try:
            with urllib.request.urlopen(f"https://files.rcsb.org/download/{pid}.{ext}.gz", timeout=60) as r:
                b = r.read()
            if len(b) > 500: p = RAW/f"{pid}.{ext}.gz"; p.write_bytes(b); return pid, p
        except Exception: continue
    return pid, None

t0=time.time()
with ThreadPoolExecutor(max_workers=16) as ex: got = list(ex.map(fetch, need))
paths = {p: f for p, f in got if f}
print(f"fetched {len(paths):,}/{len(need):,} in {(time.time()-t0)/60:.1f} min")

def ligands_and_ca(pid, path, chain_id, n_expect):
    from Bio.PDB import MMCIFParser
    try:
        parser = MMCIFParser(QUIET=True) if ".cif" in path.name else PDBParser(QUIET=True, PERMISSIVE=True)
        with gzip.open(path,"rt") as fh:
            model = next(parser.get_structure(pid, fh).get_models())
        ch = model[chain_id]
    except Exception:
        return None
    residues=[r for r in ch if r.get_resname().strip().upper() in AA3_TO_1 and "CA" in r]
    if len(residues) != n_expect: return None
    raw_ca = np.asarray([r["CA"].coord for r in residues], dtype=np.float32)

    ligs = []
    for c2 in model:
        for res in c2:
            if res.id[0] == " ": continue
            nm = res.get_resname().strip().upper()
            if nm in AA3_TO_1 or nm in NON_LIGAND: continue
            xyz = np.asarray([a.coord for a in res.get_unpacked_list() if a.element not in ("H","D")],
                             dtype=np.float32).reshape(-1,3)
            is_metal = nm in CATALYTIC_METALS
            if is_metal or len(xyz) >= MIN_LIG_ATOMS: ligs.append(xyz)
    return raw_ca, ligs

t0=time.time(); n_ok=0; n_geom_fail=0; n_no_ligand=0
for d in dataset:
    if d.pdb_id not in paths: continue
    r = ligands_and_ca(d.pdb_id, paths[d.pdb_id], d.chain_id, int(d.num_nodes))
    if r is None: continue
    raw_ca, ligs = r
    if not ligs: n_no_ligand += 1; continue

    # geometry-drift check: pairwise CA distances must match d.pos (residue ORDER, not just count)
    k = min(50, len(raw_ca))
    a = np.linalg.norm(raw_ca[:k,None]-raw_ca[None,:k], axis=-1)
    b = np.linalg.norm(d.pos.numpy()[:k,None]-d.pos.numpy()[None,:k], axis=-1)
    if float(np.abs(a-b).max()) > 1e-2:
        n_geom_fail += 1; continue

    offset = raw_ca.mean(axis=0)                       # same centering d.pos already used
    y_np = d.y.numpy().astype(bool)
    contact_centroid = raw_ca[y_np].mean(axis=0) if y_np.any() else raw_ca.mean(axis=0)
    # "primary" ligand = the one whose center of mass is nearest the labelled contact residues
    coms = [lig.mean(axis=0) for lig in ligs]
    best = int(np.argmin([np.linalg.norm(c - contact_centroid) for c in coms]))
    d.true_center = torch.from_numpy((coms[best] - offset).astype(np.float32))
    n_ok += 1

print(f"\ntrue centers recovered: {n_ok:,}/{len(dataset):,} "
      f"({(time.time()-t0)/60:.1f} min) | geometry mismatches: {n_geom_fail} | no ligand: {n_no_ligand}")

dataset = [d for d in dataset if hasattr(d, "true_center")]
splits = {s: [d for d in dataset if d.split == s] for s in ("train","val","test")}
print("splits after filtering:", {k: len(v) for k,v in splits.items()})
''')

md(r'''## Cell 2b — Model with the added localization head''')

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
    """Same TopologicalEGNN as every prior phase, plus a second head that
    predicts one pocket-center point per chain via a batch-aware softmax over
    a dedicated per-residue localization logit -- decoupled from the
    classification probability on purpose (see markdown above)."""
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
            attn = pyg_softmax(loc_logit, batch)                    # sums to 1 per graph
            weighted = attn.unsqueeze(-1) * pos                     # (N,3)
            n_graphs = int(batch.max().item()) + 1
            pred_center = torch.zeros(n_graphs, 3, device=h.device).index_add_(0, batch, weighted)
        return logits, pred_center
''')

md(r'''
## Cell 3 — Equivariance check for the NEW centroid output (before any training)

Same discipline as Phase 3/6c/11: assert the guarantee on random weights, where
a failure is unambiguous, before trusting a single number this model produces.
''')

code(r'''
@torch.no_grad()
def check_center_equivariance(n_trials=8):
    model = LocalizingEGNN(N_PRE, TOPO_DIM).to(DEVICE).eval()
    probe = dataset[0].clone()
    probe.batch = torch.zeros(probe.num_nodes, dtype=torch.long)
    probe = probe.to(DEVICE)
    _, c0 = model(probe)
    worst = 0.0
    for s in range(n_trials):
        g = torch.Generator().manual_seed(s)
        Q,_ = torch.linalg.qr(torch.randn(3,3,generator=g)); Q = (Q*torch.sign(torch.det(Q))).to(DEVICE)
        t = (torch.randn(3, generator=g) * 25.0).to(DEVICE)
        p2 = probe.clone(); p2.pos = probe.pos @ Q.T + t
        _, c1 = model(p2)
        expected = c0 @ Q.T + t
        worst = max(worst, float((c1 - expected).abs().max()))
    return worst

w = check_center_equivariance()
print(f"predicted-center equivariance: max error = {w:.3e}")
assert w < 1e-3, "LOCALIZATION HEAD IS NOT SE(3) EQUIVARIANT"
print("holds -- weighted centroid with invariant attention over equivariant positions")
''')

md(r'''## Cell 4 — Training and evaluation machinery''')

code(r'''
def run_epoch(model, loader, opt=None, lam=0.0):
    tr = opt is not None; model.train(tr)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))
    huber = nn.SmoothL1Loss(beta=4.0)                    # ~matches the 4A DCC threshold
    ps, ys, cd = [], [], []
    for b in loader:
        b = b.to(DEVICE)
        with torch.set_grad_enabled(tr):
            logits, pred_c = model(b)
            loss = crit(logits, b.y)
            if lam > 0 and pred_c is not None:
                true_c = b.true_center.view(-1, 3)
                loss = loss + lam * huber(pred_c, true_c)
                cd.append((pred_c - true_c).norm(dim=1).detach().cpu().numpy())
        if tr:
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        ps.append(torch.sigmoid(logits).detach().cpu().numpy()); ys.append(b.y.cpu().numpy())
    p, y = np.concatenate(ps), np.concatenate(ys)
    out = {"pr": average_precision_score(y, p), "roc": roc_auc_score(y, p)}
    if cd:
        d_ = np.concatenate(cd)
        out["dcc"] = float((d_ < 4.0).mean()); out["mean_center_err"] = float(d_.mean())
    return out


def fit(use_loc_head, lam, seed, epochs=40, patience=8, batch=16):
    torch.manual_seed(seed); np.random.seed(seed)
    model = LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=use_loc_head).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tl = DataLoader(splits["train"], batch_size=batch, shuffle=True)
    vl = DataLoader(splits["val"], batch_size=batch)
    best, state, bad = -1, None, 0
    for ep in range(epochs):
        run_epoch(model, tl, opt, lam)
        v = run_epoch(model, vl, lam=lam)
        sch.step()
        if v["pr"] > best:
            best, state, bad = v["pr"], {k:t.clone() for k,t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(state); model.eval()
    return model, best
''')

md(r'''## Cell 5 — Sanity dry run (catch a bug in a minute, not after a training run)''')

code(r'''
# --- verify PyG batches a per-graph (3,) attribute the way this code assumes,
# BEFORE trusting any training run on it. A wrong assumption here would not
# crash -- it would silently train against scrambled targets, which is worse
# than a crash because nothing would look wrong until the results did.
_probe_items = splits["train"][:8]
_expected = torch.stack([d.true_center for d in _probe_items])          # (8,3), known-correct order
_probe_batch = next(iter(DataLoader(_probe_items, batch_size=8)))
_got = _probe_batch.true_center.view(-1, 3)
print(f"true_center collation check: expected {tuple(_expected.shape)}, got {tuple(_got.shape)}")
assert _got.shape == _expected.shape, "COLLATION SHAPE MISMATCH -- true_center is not batching as assumed"
assert torch.allclose(_got, _expected, atol=1e-5), \
    "COLLATION ORDER/VALUE MISMATCH -- b.true_center.view(-1,3) does NOT recover per-graph centers correctly"
print("PASS: b.true_center.view(-1,3) correctly recovers per-graph targets in order")

_probe_loader = DataLoader(splits["train"][:8], batch_size=8)
_model = LocalizingEGNN(N_PRE, TOPO_DIM).to(DEVICE)
_opt = torch.optim.AdamW(_model.parameters(), lr=1e-3)
_r = run_epoch(_model, _probe_loader, _opt, lam=0.1)
print(f"dry run (8 chains, 1 step): {_r}")
assert np.isfinite(_r["pr"]) and "dcc" in _r and np.isfinite(_r["mean_center_err"])
print("gradients flow, losses finite, centroid metric present -- proceeding")
''')

md(r'''
## Cell 6 — Small lambda sweep (1 seed each), then final paired 3-seed comparison

Picks the auxiliary-loss weight on **val** DCC, never test. The final comparison
trains fresh baseline (lambda=0, i.e. classification-only, exactly the Phase 7
recipe) and auxiliary (best lambda) ensembles side by side, same data, same
architecture budget, only the loss differs.
''')

code(r'''
LAMBDAS = [0.02, 0.1, 0.5]
t0=time.time()
sweep = {}
for lam in LAMBDAS:
    m, val_pr = fit(use_loc_head=True, lam=lam, seed=0)
    v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=lam)
    sweep[lam] = {"val_pr": val_pr, "val_dcc": v.get("dcc", float("nan")),
                 "val_center_err": v.get("mean_center_err", float("nan"))}
    print(f"  lambda={lam:<5} val PR {val_pr:.4f}  val DCC {v.get('dcc',float('nan')):.4f}  "
          f"mean center err {v.get('mean_center_err',float('nan')):.2f}A  [{(time.time()-t0)/60:.1f}m]")

best_lambda = max(sweep, key=lambda k: sweep[k]["val_dcc"])
print(f"\nselected lambda (best val DCC): {best_lambda}")
''')

code(r'''
from collections import defaultdict

def heuristic_cc(ca, probs, edge_index, top_frac=0.08, min_prob=0.5, weighted=True):
    """Phase 13's WINNING heuristic (cc_weighted_tight), reused verbatim so the
    comparison below is against the best previously-known method, not a
    strawman."""
    n = len(ca)
    thresh = max(min_prob, np.quantile(probs, 1 - top_frac))
    hot = np.where(probs >= thresh)[0]
    if len(hot) == 0: return None
    hot_set = set(hot.tolist())
    adj = defaultdict(list)
    for a, b in zip(edge_index[0], edge_index[1]):
        if a in hot_set and b in hot_set: adj[a].append(b); adj[b].append(a)
    seen = set(); best_centroid, best_score = None, -np.inf
    for start in hot:
        if start in seen: continue
        stack=[start]; comp=[]
        while stack:
            u=stack.pop()
            if u in seen: continue
            seen.add(u); comp.append(u)
            stack.extend(adj[u])
        comp = np.array(comp)
        score = float(probs[comp].sum())
        if score > best_score:
            best_score = score
            w = probs[comp]; w = w / max(w.sum(), 1e-9)
            best_centroid = (ca[comp] * w[:, None]).sum(axis=0) if weighted else ca[comp].mean(axis=0)
    return best_centroid


@torch.no_grad()
def heuristic_dcc_on_test(models):
    """Per-chain (batch_size=1) so connected-components can use each chain's
    own edge_index -- ensemble-average the probabilities across seeds first."""
    hits, n = 0, 0
    for d in splits["test"]:
        db = d.clone().to(DEVICE)
        probs = np.mean([torch.sigmoid(m(db)[0]).cpu().numpy() for m in models], axis=0)
        center = heuristic_cc(d.pos.numpy(), probs, d.edge_index.numpy())
        if center is None: continue
        err = float(np.linalg.norm(center - d.true_center.numpy()))
        hits += int(err < 4.0); n += 1
    return hits / max(n, 1), n


results = {}
for tag, use_loc, lam in (("baseline_no_aux", False, 0.0), ("with_aux_loss", True, best_lambda)):
    models, vprs = [], []
    for seed in (0,1,2):
        m, vpr = fit(use_loc_head=use_loc, lam=lam, seed=seed)
        models.append(m); vprs.append(vpr)
    # ensemble-average classification probs; for centroid, average predicted points across seeds
    crit_probs, ys = [], None
    center_preds, center_true = [], []
    with torch.no_grad():
        for b in DataLoader(splits["test"], batch_size=16):
            b = b.to(DEVICE)
            seed_logits, seed_centers = [], []
            for m in models:
                lg, pc = m(b)
                seed_logits.append(torch.sigmoid(lg).cpu().numpy())
                if pc is not None: seed_centers.append(pc.cpu().numpy())
            crit_probs.append(np.mean(seed_logits, axis=0))
            if ys is None: ys = []
            ys.append(b.y.cpu().numpy())
            if seed_centers:
                center_preds.append(np.mean(seed_centers, axis=0))
                center_true.append(b.true_center.view(-1,3).cpu().numpy())
    p, y = np.concatenate(crit_probs), np.concatenate(ys)
    pr, roc = average_precision_score(y,p), roc_auc_score(y,p)
    mcc = max(matthews_corrcoef(y,(p>=t).astype(int)) for t in np.linspace(0.05,0.95,37) if (p>=t).sum())

    dcc = dca = None
    if center_preds:
        cp, ct = np.concatenate(center_preds), np.concatenate(center_true)
        d_ = np.linalg.norm(cp-ct, axis=1)
        dcc = float((d_<4.0).mean())

    heur_dcc, heur_n = heuristic_dcc_on_test(models)

    results[tag] = {"val_pr_mean": float(np.mean(vprs)), "test_pr": float(pr), "test_roc": float(roc),
                    "test_mcc": float(mcc), "test_dcc_own_center": dcc,
                    "test_dcc_best_heuristic": heur_dcc, "lambda": lam}
    print(f"[{tag}] val PR {np.mean(vprs):.4f} | test PR {pr:.4f} ROC {roc:.4f} MCC {mcc:.4f}")
    print(f"           own-centroid DCC: {dcc if dcc is not None else 'n/a (no loc head)'}")
    print(f"           best-heuristic (Phase 13 winner) DCC: {heur_dcc:.4f}  (n={heur_n})")

json.dump({"sweep": {str(k):v for k,v in sweep.items()}, "results": results, "best_lambda": best_lambda},
          open("/kaggle/working/phase14_summary.json", "w"), indent=2, default=float)
b = results["baseline_no_aux"]; a = results["with_aux_loss"]
b_heur = b["test_dcc_best_heuristic"]; a_heur = a["test_dcc_best_heuristic"]
a_own = a["test_dcc_own_center"]; pr_delta = a["test_pr"] - b["test_pr"]

print("\n" + "="*70)
print("SUMMARY: four ways to get a pocket center, same held-out test chains")
print("="*70)
print(f"  baseline model  + best heuristic   DCC {b_heur:.4f}")
print(f"  aux-loss model  + best heuristic   DCC {a_heur:.4f}")
print(f"  aux-loss model  + own centroid     DCC {a_own:.4f}")
print(f"\n  does the aux loss beat the best prior method (baseline+heuristic)? "
      f"{a_own - b_heur:+.4f} DCC")
print(f"  does the aux loss regress classification quality? {pr_delta:+.4f} PR-AUC")

print("\nsaved -> phase14_summary.json")
''')

md(r'''
---

## Reading this

Four numbers, all on the identical held-out test chains, is what makes this
comparison honest rather than suggestive:

| combination | what it tests |
|---|---|
| baseline model + best heuristic | the strongest version of the OLD approach (Phase 13's winning heuristic, applied fresh here) |
| aux-loss model + best heuristic | did the auxiliary loss sharpen the classification probabilities themselves, even judged by the old heuristic? |
| aux-loss model + own centroid | the NEW approach this notebook exists to test |

**The decisive line is `aux-loss + own centroid` minus `baseline + best
heuristic`.** That is a same-distribution, same-architecture-budget,
same-test-chains comparison of "no localization training signal + best known
post-hoc trick" against "a real localization training signal, used directly."

- **Test PR-AUC should not regress meaningfully** between baseline and
  aux-loss — if it does, the auxiliary loss is fighting the classification
  objective rather than complementing it, and lambda needs to come down (or
  the two-head design needs revisiting).
- **A clear win here** is the green light to spend the ~3 hours re-running the
  full COACH420/HOLO4K benchmark with this architecture — this result alone is
  informative but is on a different (likely easier, same-distribution-as-
  training) test set than the external benchmark, so it earns the next step
  rather than replacing it.
- **No improvement, or a PR-AUC regression**, means the auxiliary-loss idea
  itself doesn't work as designed, and is worth knowing before paying the
  external benchmark's cost again — not after.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase14_localization_training.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
