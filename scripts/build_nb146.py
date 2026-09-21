#!/usr/bin/env python3
"""Phase 146: CryptoBench 650M scale + transfer learning -- the last
missing piece to combine all 3 confirmed CryptoBench wins at ALL 3
scales (topology -- already baked into every model; transfer learning
-- confirmed +0.0218 val PR at 35M; cross-scale ensemble). 35M-transfer
has all 8 seeds (Phase122+128), 3B-transfer has all 8 seeds (Phase129);
this trains 650M-transfer's 8 seeds, each from a DIFFERENT main-pipeline
650M checkpoint (Phase145, all 8 trained fresh in one GPU kernel after
the CPU-chunking saga -- confirmed dimensionally identical: 1341 =
21+6+1280+34, matching Phase45's CryptoBench-650M blob exactly)."""
import json
from pathlib import Path

nb46 = json.load(open("/Users/shivenkhurana/bioproj/phase46_cryptobench_train_eval_esm650m.ipynb"))
c46 = nb46["cells"]
s46 = lambda i: (c46[i]["source"] if isinstance(c46[i]["source"], str) else "".join(c46[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 146 — CryptoBench 650M + Transfer Learning (8 seeds)

Completes transfer learning at all 3 scales. Uses the main pipeline's
650M COACH420 checkpoints (Phase145, freshly trained after the GPU
650M-embedding breakthrough) as init -- dimensionally identical
(1341 = 21+6+1280+34). Each of the 8 fine-tune seeds uses a DIFFERENT
pretrained seed to avoid conflating "does transfer help" with "was one
seed lucky" (same discipline as Phase128/129).
''')

md(r'''## Cell 1 — Install dependencies''')
code(s46(2))

md(r'''## Cell 2 — Setup, load Phase 45's CryptoBench-650M blob (identical to Phase46/47)''')
code(s46(3))

md(r'''## Cell 3 — Model (identical to every prior localization phase)''')
code(s46(5))

md(r'''## Cell 4 — Training machinery: fit() optionally loads a pretrained state_dict before training''')
code(r'''
def run_epoch(model, loader, opt=None, lam=0.0):
    tr = opt is not None; model.train(tr)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))
    huber = nn.SmoothL1Loss(beta=4.0)
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

md(r'''## Cell 5 — Train all 8 seeds, each transfer-learned from a DIFFERENT main-pipeline 650M checkpoint''')
code(r'''
PRETRAINED_FILES = [f"clean_deploy_esm650m_seed{s}.pt" for s in range(8)]   # Phase145, all 8 seeds

BEST_LAMBDA = 0.1
results = []
t0 = time.time()
for seed in range(8):
    pretrained = locate(PRETRAINED_FILES[seed])
    m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed, pretrained_path=pretrained)
    v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
    torch.save(m.state_dict(), WORK/f"cache/cryptobench_esm650m_transfer_seed{seed}.pt")
    val_dcc = v.get("dcc", float("nan"))
    results.append(dict(seed=seed, val_pr=val_pr, val_dcc=val_dcc))
    print(f"  650M-transfer seed {seed} (pretrained: {pretrained.name}): val PR {val_pr:.4f}  val DCC {val_dcc:.4f}  [{(time.time()-t0)/60:.1f}m]")

all_pr = [r["val_pr"] for r in results]
all_dcc = [r["val_dcc"] for r in results]
print(f"\n--- 650M-transfer, 8-seed summary ---")
print(f"val PR:  mean {np.mean(all_pr):.4f}  std {np.std(all_pr):.4f}")
print(f"val DCC: mean {np.mean(all_dcc):.4f}  std {np.std(all_dcc):.4f}")
print(f"\nfor reference, 650M from-scratch (Phase46/47): see phase47 8-seed results")
json.dump({"seeds": results, "all_pr": all_pr, "all_dcc": all_dcc},
          open(WORK/"phase146_esm650m_transfer_8seed_results.json", "w"), indent=2)
print("\nsaved -> phase146_esm650m_transfer_8seed_results.json")
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase146_cryptobench_esm650m_transfer.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
