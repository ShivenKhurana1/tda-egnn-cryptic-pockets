#!/usr/bin/env python3
"""Phase 119: CryptoBench -- does MASKING ambiguous residues (non-cryptic
pocket, but not cryptic) out of the main loss help, where the AUXILIARY-
HEAD approach (Phase118) clearly hurt (monotonic decline as lam_aux rose:
0.3592 -> 0.3557 -> 0.3476)? Different mechanism: instead of asking the
model to ALSO predict "any pocket" (a competing task that diluted
gradients), this simply REMOVES residues that are pocket-like-but-not-
cryptic from the main classification loss entirely -- they may be acting
as confusing forced-negative examples (label 0, but likely to look like
true positives in feature space) that hurt the decision boundary. Uses
the same Phase117 blob (already has y_aux) and the same seed=0/split as
Phase118 for a direct, comparable readout."""
import json
from pathlib import Path

nb118 = json.load(open("/Users/shivenkhurana/bioproj/phase118_cryptobench_train_aux_sweep.ipynb"))
c118 = nb118["cells"]
s118 = lambda i: c118[i]["source"]

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 119 - CryptoBench: Mask Ambiguous Residues Instead of an Aux Head

Phase118's auxiliary-head approach clearly hurt (monotonic decline as
lam_aux rose). Different mechanism this time: residues that are
pocket-like-but-not-cryptic (`y_aux=1, y=0`) may be confusing forced
negatives in the main BCE loss -- they likely look like true positives in
feature space but get penalized as if they were clearly not. This masks
them OUT of the main loss entirely (no gradient either way) instead of
asking the model to also predict them. Trains baseline (unmasked,
reproducing Phase118's lam_aux=0 exactly) and the masked variant in the
SAME run, seed=0, for direct comparison.
''')

md(r'''## Cell 1 - Install dependencies''')
code(s118(2))

md(r'''## Cell 2 - Setup, load Phase 117's aux-label CryptoBench blob (identical to Phase118)''')
code(s118(3))

md(r'''## Cell 3 - Model (identical to Phase118; use_aux_head not needed for masking, only y_aux is used to build the loss mask)''')
code(s118(5))

md(r'''## Cell 4 - Training machinery: run_epoch masks the main loss when mask_ambiguous=True (keeps only residues where y_aux==y: clear negatives [0,0] or clear positives [1,1], since y_aux is a strict superset of y). Validation/test metrics always computed over ALL residues (unmasked) for a fair, comparable readout.''')
code(r'''
def run_epoch(model, loader, opt=None, lam=0.0, mask_ambiguous=False):
    tr = opt is not None; model.train(tr)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE),
                                 reduction="none" if mask_ambiguous else "mean")
    huber = nn.SmoothL1Loss(beta=4.0)
    ps, ys, cd = [], [], []
    for b in loader:
        b = b.to(DEVICE)
        with torch.set_grad_enabled(tr):
            logits, pred_c, _ = model(b)
            if mask_ambiguous:
                raw = crit(logits, b.y)
                mask = (b.y_aux == b.y).float()   # drop y_aux=1,y=0 (ambiguous non-cryptic pocket) residues
                loss = (raw * mask).sum() / mask.sum().clamp(min=1)
            else:
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


def fit(use_loc_head, lam, seed, mask_ambiguous=False, epochs=40, patience=8, batch=16):
    torch.manual_seed(seed); np.random.seed(seed)
    model = LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=use_loc_head, use_aux_head=False).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tl = DataLoader(splits["train"], batch_size=batch, shuffle=True)
    vl = DataLoader(splits["val"], batch_size=batch)
    best, state, bad = -1, None, 0
    for ep in range(epochs):
        run_epoch(model, tl, opt, lam, mask_ambiguous)
        v = run_epoch(model, vl, lam=lam, mask_ambiguous=mask_ambiguous)
        sch.step()
        if v["pr"] > best:
            best, state, bad = v["pr"], {k:t.clone() for k,t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(state); model.eval()
    return model, best
''')

md(r'''## Cell 5 - Train baseline (unmasked, reproduces Phase118 lam_aux=0) and the masked variant, same seed=0, same run''')
code(r'''
BEST_LAMBDA = 0.1   # localization-loss weight, unchanged from Phase42/43/118
results = []
t0 = time.time()
for mask_ambiguous, name in [(False, "baseline (unmasked)"), (True, "masked (drop ambiguous non-cryptic-pocket residues)")]:
    m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, mask_ambiguous=mask_ambiguous, seed=0)
    v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA, mask_ambiguous=mask_ambiguous)
    val_dcc = v.get("dcc", float("nan"))
    torch.save(m.state_dict(), WORK/f"cache/cryptobench_mask{mask_ambiguous}_seed0.pt")
    results.append(dict(mask_ambiguous=mask_ambiguous, name=name, val_pr=val_pr, val_dcc=val_dcc))
    print(f"  {name:<55} val PR {val_pr:.4f}  val DCC {val_dcc:.4f}  [{(time.time()-t0)/60:.1f}m]")

print(f"\n--- masking comparison (single seed=0, same run) ---")
baseline_pr = next(r["val_pr"] for r in results if not r["mask_ambiguous"])
for r in results:
    print(f"  {r['name']:<55} val PR {r['val_pr']:.4f}  ({r['val_pr']-baseline_pr:+.4f} vs baseline)  val DCC {r['val_dcc']:.4f}")
print(f"\nfor reference, Phase118's lam_aux=0 baseline (same seed/split): val PR 0.3592, val DCC 0.1833")
json.dump(results, open(WORK/"phase119_mask_results.json", "w"), indent=2)
print("\nsaved -> phase119_mask_results.json")
''')

md(r'''
---

## Reading this

- **Masked beats baseline by a real margin (outside typical single-seed
  noise, roughly >0.01-0.015 val PR)** -- worth a multi-seed confirmation
  before trusting it, same discipline as every other lever this project
  tests.
- **Masked ties or barely beats baseline** -- another negative-adjacent
  result; the ambiguous residues probably weren't the actual bottleneck.
- **Masked is WORSE** -- would suggest those "ambiguous" residues were
  actually informative negative examples the model needs, not noise.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase119_cryptobench_mask_ambiguous.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
