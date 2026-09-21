#!/usr/bin/env python3
"""8-seed extension of Phase122's transfer-learning ablation: trains ONE
arm (scratch or transfer) across seeds 1-7 (seed 0 already banked from
Phase122: scratch=0.3537, transfer=0.3700). For the transfer arm, tries
5 DIFFERENT Phase32 pretrained seeds (3,4,5,6,7) instead of only seed3
repeated -- avoids conflating "does transfer learning help" with "was
seed3 specifically a lucky pretrained checkpoint" (the seeds 0,1,2
weren't available in Phase32's own kernel_sources at build time; using
its extension seeds 3-7 covers 5 of the 7 new fine-tune seeds, with the
last 2 reusing seed3/seed4 again since only 5 distinct pretrained
checkpoints are available from Phase32's own kernel_sources).
Usage: python3 build_nb_transfer_arm.py <scratch|transfer> <PHASE_NUM>"""
import json, sys
from pathlib import Path

ARM = sys.argv[1]
assert ARM in ("scratch", "transfer")
PHASE_NUM = sys.argv[2]

nb118 = json.load(open("/Users/shivenkhurana/bioproj/phase118_cryptobench_train_aux_sweep.ipynb"))
c118 = nb118["cells"]
s118 = lambda i: c118[i]["source"]

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(f'''
# Phase {PHASE_NUM} — Transfer-Learning 8-Seed Extension: {ARM} arm, seeds 1-7

Extends Phase122's single-seed transfer-learning comparison (scratch:
val PR 0.3537, transfer: val PR 0.3700, seed 0) to a properly-powered
8-seed comparison. This notebook trains ONLY the "{ARM}" arm across
seeds 1-7.
''')

md(r'''## Cell 1 — Install dependencies''')
code(s118(2))

md(r'''## Cell 2 — Setup, load Phase 117's blob (identical to Phase118/119/121/122)''')
code(s118(3))

md(r'''## Cell 3 — Model (identical to Phase118/119/121/122)''')
code(s118(5))

md(r'''## Cell 4 — Training machinery (fit() optionally loads pretrained weights)''')
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

md(f'''## Cell 5 — Train "{ARM}" arm, seeds 1-7''')
if ARM == "scratch":
    loop_body = '''for seed in range(1, 8):
    m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed, pretrained_path=None)'''
else:
    # 5 distinct Phase32 pretrained checkpoints available (seeds 3-7);
    # fine-tune seeds 1-7 each load a DIFFERENT pretrained seed (cycling)
    # to avoid conflating "transfer helps" with "seed3 was a lucky pretrain"
    loop_body = '''PRETRAINED_SEEDS = [3, 4, 5, 6, 7, 3, 4]   # cycle through Phase32's 5 available checkpoints
for seed, pt_seed in zip(range(1, 8), PRETRAINED_SEEDS):
    pretrained = locate(f"clean_deploy_scaled_seed{pt_seed}.pt")
    m, val_pr = fit(use_loc_head=True, lam=BEST_LAMBDA, seed=seed, pretrained_path=pretrained)'''

code(f'''
BEST_LAMBDA = 0.1
results = []
t0 = time.time()
{loop_body}
    v = run_epoch(m, DataLoader(splits["val"], batch_size=16), lam=BEST_LAMBDA)
    torch.save(m.state_dict(), WORK/f"cache/cryptobench_{ARM}_seed{{seed}}.pt")
    val_dcc = v.get("dcc", float("nan"))
    results.append(dict(seed=seed, val_pr=val_pr, val_dcc=val_dcc))
    print(f"  {ARM} seed {{seed}}: val PR {{val_pr:.4f}}  val DCC {{val_dcc:.4f}}  [{{(time.time()-t0)/60:.1f}}m]")

seed0_pr = {'0.3537' if ARM=='scratch' else '0.3700'}
seed0_dcc = {'0.1833' if ARM=='scratch' else '0.2500'}
all_pr = [seed0_pr] + [r["val_pr"] for r in results]
all_dcc = [seed0_dcc] + [r["val_dcc"] for r in results]
print(f"\\n--- {ARM} arm, full 8-seed summary (seed0 from Phase122 + seeds1-7 here) ---")
print(f"val PR:  mean {{np.mean(all_pr):.4f}}  std {{np.std(all_pr):.4f}}  values {{[round(x,4) for x in all_pr]}}")
print(f"val DCC: mean {{np.mean(all_dcc):.4f}}  std {{np.std(all_dcc):.4f}}  values {{[round(x,4) for x in all_dcc]}}")
json.dump({{"seed0_pr": seed0_pr, "seed0_dcc": seed0_dcc, "seeds_1to7": results,
           "all_pr": all_pr, "all_dcc": all_dcc}},
          open(WORK/f"phase{PHASE_NUM}_{ARM}_8seed_results.json", "w"), indent=2)
print(f"\\nsaved -> phase{PHASE_NUM}_{ARM}_8seed_results.json")
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path(f"/Users/shivenkhurana/bioproj/phase{PHASE_NUM}_transfer_{ARM}_8seed.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
