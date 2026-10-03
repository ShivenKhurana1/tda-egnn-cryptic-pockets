#!/usr/bin/env python3
"""Phase 122: CryptoBench transfer learning from the main pipeline.
CryptoBench has only ~700 training chains; the main COACH420/HOLO4K
pipeline has ~7,181 (round-3 scale) -- 10x more data on a related task
(protein binding-site prediction, same architecture: LocalizingEGNN,
h=128/L=4, IDENTICAL 541-dim node width at 35M ESM scale, confirmed by
inspecting Phase32's checkpoint directly: embed.0.weight is [128, 541],
byte-for-byte matching CryptoBench's own 35M-scale node width). Loads
Phase32's round-3 COACH420 checkpoint (seed3) as INITIALIZATION, then
fine-tunes fully on CryptoBench's own loss/data -- standard transfer
learning. Compares against from-scratch, same seed=0, same run."""
import json
from pathlib import Path

nb118 = json.load(open("/Users/shivenkhurana/bioproj/phase118_cryptobench_train_aux_sweep.ipynb"))
c118 = nb118["cells"]
s118 = lambda i: c118[i]["source"]

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 122 - CryptoBench: Transfer Learning from the Main Pipeline

CryptoBench has ~700 training chains vs the main pipeline's ~7,181
(round-3) -- a 10x data gap on a related task (binding-site prediction,
identical architecture and 541-dim node width at 35M ESM scale, verified
by inspecting Phase32's checkpoint directly). Loads Phase32's round-3
COACH420 seed3 checkpoint as INITIALIZATION, then fine-tunes fully on
CryptoBench's own loss/data. Compares against from-scratch, same seed=0,
same run.
''')

md(r'''## Cell 1 - Install dependencies''')
code(s118(2))

md(r'''## Cell 2 - Setup, load Phase 117's blob (identical to Phase118/119/121)''')
code(s118(3))

md(r'''## Cell 3 - Model (identical to Phase118/119/121)''')
code(s118(5))

md(r'''## Cell 4 - Training machinery: fit() optionally loads a pretrained state_dict before training (transfer learning) instead of random init''')
code(r'''
def run_epoch(model, loader, opt=None, lam=0.0):
    tr = opt is not None; model.train(tr)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))
    huber = nn.SmoothL1Loss(beta=4.0)
    ps, ys, cd = [], [], []
    for b in loader:
        b = b.to(DEVICE)
        with torch.set_grad_enabled(tr):
            logits, pred_c, _ = model(b)
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


def fit(use_loc_head, lam, seed, pretrained_path=None, epochs=40, patience=8, batch=16):
    torch.manual_seed(seed); np.random.seed(seed)
    model = LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=use_loc_head, use_aux_head=False).to(DEVICE)
    if pretrained_path is not None:
        state = torch.load(pretrained_path, map_location=DEVICE)
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not missing and not unexpected, f"architecture mismatch: missing={missing} unexpected={unexpected}"
        print(f"    loaded pretrained weights from {pretrained_path.name} (strict match, 0 missing/unexpected keys)")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tl = DataLoader(splits["train"], batch_size=batch, shuffle=True)
    vl = DataLoader(splits["val"], batch_size=batch)
    best, state_best, bad = -1, None, 0
    for ep in range(epochs):
        run_epoch(model, tl, opt, lam)
        v = run_epoch(model, vl, lam=lam)
        sch.step()
        if v["pr"] > best:
            best, state_best, bad = v["pr"], {k:t.clone() for k,t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(state_best); model.eval()
    return model, best
''')

md(r'''## Cell 5 - Train from-scratch first (data/model pristine), then transfer-learned (Phase32 seed3 init), same seed=0, same run''')
code(r'''
BEST_LAMBDA = 0.1
results = []
t0 = time.time()

m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=0, pretrained_path=None)
v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
torch.save(m.state_dict(), WORK/"cache/cryptobench_scratch_seed0.pt")
results.append(dict(arm="from scratch", val_pr=val_pr, val_dcc=v.get("dcc", float("nan"))))
print(f"  from scratch          seed0: val PR {val_pr:.4f}  val DCC {v.get('dcc', float('nan')):.4f}  [{(time.time()-t0)/60:.1f}m]")

pretrained = locate("clean_deploy_scaled_seed3.pt")
m2, val_pr2 = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=0, pretrained_path=pretrained)
v2 = run_epoch(m2, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
torch.save(m2.state_dict(), WORK/"cache/cryptobench_transfer_seed0.pt")
results.append(dict(arm="transfer-learned (Phase32 seed3 init)", val_pr=val_pr2, val_dcc=v2.get("dcc", float("nan"))))
print(f"  transfer-learned      seed0: val PR {val_pr2:.4f}  val DCC {v2.get('dcc', float('nan')):.4f}  [{(time.time()-t0)/60:.1f}m]")

print(f"\n--- transfer-learning comparison (single seed=0, same run) ---")
scratch_pr = results[0]["val_pr"]
for r in results:
    print(f"  {r['arm']:<40} val PR {r['val_pr']:.4f}  ({r['val_pr']-scratch_pr:+.4f} vs from-scratch)  val DCC {r['val_dcc']:.4f}")
json.dump(results, open(WORK/"phase122_transfer_results.json", "w"), indent=2)
print("\nsaved -> phase122_transfer_results.json")
''')

md(r'''
---

## Reading this

- **transfer-learned clearly beats from-scratch** (gap outside typical
  single-seed noise) -- the main pipeline's larger training pool gives
  CryptoBench useful prior knowledge despite the different task
  (apo-visible vs apo-invisible pockets); worth an 8-seed confirmation
  and worth trying pretraining on MULTIPLE main-pipeline seeds/checking
  which pretrained seed transfers best.
- **Close/tied** -- the tasks are different enough (visible vs cryptic
  pockets) that the pretrained features don't transfer usefully; from-
  scratch on task-specific data is just as good.
- **transfer-learned is WORSE** -- would suggest negative transfer: the
  main pipeline's learned features are actively mismatched for cryptic-
  pocket detection, worth understanding before trying this direction
  again.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase122_cryptobench_transfer.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
