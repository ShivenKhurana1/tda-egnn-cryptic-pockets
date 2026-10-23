#!/usr/bin/env python3
"""Phase 299: test-set measurement of the CV-selected stack3 = ESM2-3B input + ESM2-650M + ProtT5 + SaProt branches (selected on the
Phase 298 CV instrument; test untouched until now). Phase 297 protocol: Phase 291 recipe, early stopping on the 60-chain validation
set, 8 seeds; base and stack2 (Phase 297's model) re-run in the same kernel for pairing. Embeddings as in Phase 298 (the Ankh pass still
runs there but no arm uses it).
usage: build_nb299.py <output_root>"""
import json, subprocess, sys, tempfile
from pathlib import Path

ROOT_OUT = Path(sys.argv[1]).resolve(); SLUG = "299"
HERE = Path(__file__).resolve().parent
tmp = Path(tempfile.mkdtemp())
subprocess.run([sys.executable, str(HERE.parent / "phase298" / "build_nb298.py"), str(tmp)], check=True, capture_output=True)
nb = json.load(open(tmp / "298" / "phase298_ankh_saprot_cv.ipynb"))
cells = nb["cells"]
src = lambda i: cells[i]["source"] if isinstance(cells[i]["source"], str) else "".join(cells[i]["source"])
find = lambda key: next(i for i in range(len(cells)) if key in src(i))
cells[0]["source"] = "# Phase 299 - test-set measurement: base vs stack2 (650M+ProtT5) vs stack3 (650M+ProtT5+SaProt), early stopping on val\n"
i = find('ARMS = [("stack2"')
cells[i]["source"] = r'''
KEEP_SIDE = list(range(TOPO_DIM)); m_ = torch.zeros(SIDE_DIM); m_[KEEP_SIDE] = 1.0
for d in dataset: d.x[:, N_PRE:] = d.x[:, N_PRE:] * m_

def val_pr(model):
    ps, ys = [], []
    with torch.no_grad():
        for b in DataLoader(splits["val"], batch_size=16):
            b = b.to(DEVICE); lg, _ = model(b); ps.append(torch.sigmoid(lg).cpu().numpy()); ys.append(b.y.cpu().numpy())
    return average_precision_score(np.concatenate(ys), np.concatenate(ps))

def fit_es(seed, pretrained_path, dims, epochs=40, patience=8, lam=0.1, batch=16):
    """Phase 129/291/297 fit semantics: AdamW 1e-3/1e-4, cosine over 40 epochs, keep the best val-PR state, stop after 8 bad epochs."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = LocalizingEGNN(N_PRE, SIDE_DIM, use_loc_head=True, **dims).to(DEVICE)
    missing, unexpected = model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE), strict=False)
    assert not [k for k in missing if not k.startswith(("plm_proj", "plm2_proj", "plm3_proj", "plm4_proj"))] and not unexpected
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tl = DataLoader(splits["train"], batch_size=batch, shuffle=True); best, bad, state, ep_used = -1, 0, None, 0
    for ep in range(epochs):
        run_epoch(model, tl, opt, lam, "bce", 1.0); sch.step(); model.eval(); v = val_pr(model); ep_used = ep + 1
        if v > best: best, bad, state = v, 0, {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(state); return model.eval(), best, ep_used

@torch.no_grad()
def predict_split(model, split):
    out = {}
    for d in splits[split]:
        b = d.clone().to(DEVICE); lg, pc = model(b)
        out[(str(d.pdb_id).lower(), str(d.chain_id))] = dict(p=torch.sigmoid(lg).cpu().numpy().astype(np.float32), c=pc.cpu().numpy()[0],
            y=d.y.numpy().astype(np.int8), y_main=d.y_main.numpy().astype(np.int8), tc=d.true_center.numpy())
    return out

ARMS = [("base", {}), ("stack2", dict(plm_dim=1280, plm2_dim=1024)), ("stack3", dict(plm_dim=1280, plm2_dim=1024, plm4_dim=1280))]
T0 = time.time(); ALL = {}
for name, dims in ARMS:
    ALL[name] = {"seeds": {}}
    for seed in range(8):
        model, v, ep = fit_es(seed, locate(f"pretrain266_full_seed{seed}.pt"), dims)
        ALL[name]["seeds"][seed] = {"val": predict_split(model, "val"), "test": predict_split(model, "test"), "val_pr": float(v), "epochs": ep}
        torch.save(ALL, WORK / "phase299_test_preds.pt")
        print(f"  [{name}] seed {seed}: val PR {v:.4f} ({ep} ep)  [{(time.time()-T0)/60:.1f}m]", flush=True)
print("done")
'''
d = ROOT_OUT / SLUG; d.mkdir(parents=True, exist_ok=True)
o = d / "phase299_stack3_test.ipynb"; o.write_text(json.dumps(nb, indent=1) + "\n")
meta = json.load(open(tmp / "298" / "kernel-metadata.json"))
meta.update(id="blackpearls1/phase299-stack3-test", title="Phase299 Stack3 Test", code_file=o.name)
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote", o)
