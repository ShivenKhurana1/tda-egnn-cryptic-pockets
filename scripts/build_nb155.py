#!/usr/bin/env python3
"""Phase 155: CryptoBench 3B-transfer with FOCAL LOSS instead of
BCE+pos_weight, 8 seeds. Tests whether a loss designed for extreme class
imbalance (here ~4.7% positive) beats a fixed positive-class weight on
CryptoBench's own residue classification objective. Otherwise identical
to Phase129 (same 8 pretrained main-pipeline 3B checkpoints, same
architecture, same training protocol) for a clean, paired-by-seed
comparison against Phase129's existing 8-seed BCE result
(val PR mean 0.4023, std 0.0102; test AUC 0.9111, MCC 0.4664)."""
import json
from pathlib import Path

nb129 = json.load(open("/Users/shivenkhurana/bioproj/phase129_cryptobench_esm3b_transfer.ipynb"))
c129 = nb129["cells"]
s129 = lambda i: (c129[i]["source"] if isinstance(c129[i]["source"], str) else "".join(c129[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 155 - CryptoBench 3B-Transfer with Focal Loss (8 seeds)

Identical to Phase129 (same 8 pretrained main-pipeline 3B checkpoints,
same architecture, same protocol) except the loss function: focal loss
(alpha=0.9, gamma=2.0 -- alpha up-weights the rare positive class,
consistent in spirit with BCE's pos_weight=20.73 but with the added
gamma-modulated down-weighting of easy, already-well-classified examples
that plain pos_weight lacks) replaces BCEWithLogitsLoss+pos_weight.
Paired-by-seed comparison against Phase129's existing 8-seed BCE result
(val PR mean 0.4023 std 0.0102; test AUC 0.9111, MCC 0.4664).
''')

md(r'''## Cell 1 - Install dependencies''')
code(s129(2))

md(r'''## Cell 2 - Setup, load Phase 48's CryptoBench-3B blob (identical to Phase129)''')
code(s129(4))

md(r'''## Cell 3 - Model (identical to every prior localization phase)''')
code(s129(6))

md(r'''## Cell 4 - Training machinery: focal loss replaces BCE+pos_weight; fit() optionally loads a pretrained state_dict''')
code(r'''
def focal_loss_with_logits(logits, targets, alpha=0.9, gamma=2.0):
    """Standard binary focal loss (Lin et al. 2017), computed from logits
    for numerical stability. alpha up-weights the positive class (rare,
    ~4.7% here); gamma down-weights already-easy (high-confidence-correct)
    examples so training focuses on hard cases, which pos_weight alone
    does not do."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t).pow(gamma) * bce
    return loss.mean()


def run_epoch(model, loader, opt=None, lam=0.0):
    tr = opt is not None; model.train(tr)
    huber = nn.SmoothL1Loss(beta=4.0)
    ps, ys, cd = [], [], []
    for b in loader:
        b = b.to(DEVICE)
        with torch.set_grad_enabled(tr):
            logits, pred_c = model(b)
            loss = focal_loss_with_logits(logits, b.y, alpha=0.9, gamma=2.0)
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
    model = LocalizingEGNN(N_PRE, TOPO_DIM, use_loc_head=use_loc_head).to(DEVICE)
    if pretrained_path is not None:
        state = torch.load(pretrained_path, map_location=DEVICE)
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not missing and not unexpected, f"architecture mismatch: missing={missing} unexpected={unexpected}"
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

md(r'''## Cell 5 - Train all 8 seeds, each transfer-learned from a DIFFERENT main-pipeline 3B checkpoint, with focal loss''')
code(r'''
PRETRAINED_FILES = [f"clean_deploy_esm3b_seed{s}.pt" for s in range(8)]   # Phase97(0-2)+salvage(3-6)+Phase107(7)

BEST_LAMBDA = 0.1
results = []
t0 = time.time()
for seed in range(8):
    pretrained = locate(PRETRAINED_FILES[seed])
    m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed, pretrained_path=pretrained)
    v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
    torch.save(m.state_dict(), WORK/f"cache/cryptobench_esm3b_focal_seed{seed}.pt")
    val_dcc = v.get("dcc", float("nan"))
    results.append(dict(seed=seed, val_pr=val_pr, val_dcc=val_dcc))
    print(f"  3B-focal seed {seed} (pretrained: {pretrained.name}): val PR {val_pr:.4f}  val DCC {val_dcc:.4f}  [{(time.time()-t0)/60:.1f}m]")

all_pr = [r["val_pr"] for r in results]
all_dcc = [r["val_dcc"] for r in results]
print(f"\n--- 3B-focal, 8-seed summary ---")
print(f"val PR:  mean {np.mean(all_pr):.4f}  std {np.std(all_pr):.4f}  values {[round(x,4) for x in all_pr]}")
print(f"val DCC: mean {np.mean(all_dcc):.4f}  std {np.std(all_dcc):.4f}")
print(f"\nfor reference, Phase129's 3B-transfer with BCE+pos_weight: val PR mean 0.4023 std 0.0102")
json.dump({"seeds": results, "all_pr": all_pr, "all_dcc": all_dcc},
          open(WORK/"phase155_esm3b_focal_8seed_results.json", "w"), indent=2)
print("\nsaved -> phase155_esm3b_focal_8seed_results.json")
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase155_cryptobench_esm3b_focal.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
