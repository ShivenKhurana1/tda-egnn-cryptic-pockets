#!/usr/bin/env python3
"""Phase 306: a DEPLOYMENT-MATCHED cross-validation instrument. Phase 305 showed a lever confirmed on the Phase 295 CV (fixed 12 epochs)
vanishing under the deployment protocol (early stopping on validation pooled PR). Here each CV fold trains exactly like deployment: from
the three training folds, a random 8% of chains (seeded by seed and fold; the deployment val split is 60/760 = 7.9%) is held out as an inner
early-stopping set; up to 40 epochs, cosine schedule over 40, best inner-val pooled-PR state kept, patience 8. The held-out fold is then
predicted, so all 760 chains are still evaluated once per seed and the test set stays untouched.
INSTRUMENT VALIDATION (pre-registered): arms stack2 (ESM2-3B + 650M + ProtT5), stack3 (+SaProt), stack3_hot_studemb, 8 seeds x 4 folds,
one kernel. The instrument agrees with deployment if (a) stack3 vs stack2 is positive on pooled PR and per-protein AUC (on test: pooled PR
+0.012, ROC +0.011 significant), and (b) stack3_hot_studemb vs stack3 is not a confirmed gain on per-protein AUC (on test: +0.0003).
usage: build_nb306.py <output_root>"""
import json, subprocess, sys, tempfile
from pathlib import Path
ROOT_OUT = Path(sys.argv[1]).resolve(); SLUG = "306"
HERE = Path(__file__).resolve().parent
tmp = Path(tempfile.mkdtemp())
subprocess.run([sys.executable, str(HERE.parent / "phase303" / "build_nb303.py"), str(tmp)], check=True, capture_output=True)
nb = json.load(open(tmp / "303" / "phase303_hot_studemb_cv.ipynb")); cells = nb["cells"]
src = lambda i: cells[i]["source"] if isinstance(cells[i]["source"], str) else "".join(cells[i]["source"])
cells[0]["source"] = "# Phase 306 - deployment-matched CV instrument (inner early stopping per fold), validated against known test outcomes\n"
i = next(i for i in range(len(cells)) if "ARMS = [(" in src(i))
cells[i]["source"] = r'''
KEEP_SIDE = list(range(TOPO_DIM)); m_ = torch.zeros(SIDE_DIM); m_[KEEP_SIDE] = 1.0
for d in dataset: d.x[:, N_PRE:] = d.x[:, N_PRE:] * m_
AUX_W = 0.3

def pooled_pr(model, chains):
    ps, ys = [], []
    with torch.no_grad():
        for b in DataLoader(chains, batch_size=16):
            b = b.to(DEVICE); lg, _ = model(b); ps.append(torch.sigmoid(lg).cpu().numpy()); ys.append(b.y.cpu().numpy())
    return average_precision_score(np.concatenate(ys), np.concatenate(ps))

def fit_es(train_set, es_set, seed, pretrained_path, dims, flags, epochs=40, patience=8, lam=0.1, batch=16):
    """Deployment fit semantics (Phases 129/291/299/305) with an inner early-stopping set."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = LocalizingEGNN(N_PRE, SIDE_DIM, use_loc_head=True, **dims, **flags).to(DEVICE)
    missing, unexpected = model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE), strict=False)
    assert not [k for k in missing if not k.startswith(("plm_proj", "plm2_proj", "plm4_proj", "hemb_head", "hot_proj"))] and not unexpected
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tl = DataLoader(train_set, batch_size=batch, shuffle=True); best, bad, state, ep_used = -1, 0, None, 0
    for ep in range(epochs):
        run_epoch(model, tl, opt, lam, "bce", 1.0); sch.step(); model.eval(); v = pooled_pr(model, es_set); ep_used = ep + 1
        if v > best: best, bad, state = v, 0, {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(state); return model.eval(), ep_used

ARMS = [("stack2", dict(plm_dim=1280, plm2_dim=1024), {}),
        ("stack3", dict(plm_dim=1280, plm2_dim=1024, plm4_dim=1280), {}),
        ("stack3_hot_studemb", dict(plm_dim=1280, plm2_dim=1024, plm4_dim=1280), dict(hot=True, auxemb=True))]
T0 = time.time(); BUDGET = 11.0 * 3600; ALL = {}; EPOCHS = {}
for name, dims, flags in ARMS:
    for seed in range(8):
        if time.time() - T0 > BUDGET: print(f"budget reached before {name} seed {seed}"); break
        ck = locate(f"pretrain266_full_seed{seed}.pt"); preds = {}
        for f in range(4):
            pool = [d for d in CV if d.fold != f]
            rs = np.random.RandomState(1000 * seed + f); idx = rs.permutation(len(pool)); n_es = int(round(0.08 * len(pool)))
            es_set = [pool[j] for j in idx[:n_es]]; train_set = [pool[j] for j in idx[n_es:]]
            model, ep = fit_es(train_set, es_set, seed, ck, dims, flags); EPOCHS.setdefault(name, []).append(ep)
            preds.update(predict(model, [d for d in CV if d.fold == f]))
        ALL.setdefault(name, {})[seed] = preds
        torch.save({"preds": ALL, "epochs": EPOCHS}, WORK / "phase306_cv_preds.pt")
        pp = np.concatenate([v["p"] for v in preds.values()]); yy = np.concatenate([v["y"] for v in preds.values()])
        print(f"  seed {seed} [{name}] held-out PR {average_precision_score(yy, pp):.4f}  epochs {EPOCHS[name][-4:]}  [{(time.time()-T0)/60:.1f}m]", flush=True)
print("done")
'''
d = ROOT_OUT / SLUG; d.mkdir(parents=True, exist_ok=True)
o = d / "phase306_deploy_cv.ipynb"; o.write_text(json.dumps(nb, indent=1) + "\n")
meta = json.load(open(tmp / "303" / "kernel-metadata.json")); meta.update(id="blackpearls1/phase306-deploy-cv", title="Phase306 Deploy Cv", code_file=o.name)
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n"); print("wrote", o)
