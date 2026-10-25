#!/usr/bin/env python3
"""Phase 305: test-set measurement of stack3_hot_studemb (confirmed on the Phase 304 CV) vs stack3, Phase 299 protocol: early stopping on
the 60-chain validation set, 8 seeds, both arms in one kernel. The holo-embedding loss uses training chains only; predictions read only the
apo structure, its language-model embeddings and its hotspot map.
usage: build_nb305.py <output_root>"""
import json, subprocess, sys, tempfile
from pathlib import Path
ROOT_OUT = Path(sys.argv[1]).resolve(); SLUG = "305"
HERE = Path(__file__).resolve().parent
tmp = Path(tempfile.mkdtemp())
subprocess.run([sys.executable, str(HERE.parent / "phase303" / "build_nb303.py"), str(tmp)], check=True, capture_output=True)
nb = json.load(open(tmp / "303" / "phase303_hot_studemb_cv.ipynb")); cells = nb["cells"]
src = lambda i: cells[i]["source"] if isinstance(cells[i]["source"], str) else "".join(cells[i]["source"])
cells[0]["source"] = "# Phase 305 - test-set measurement: stack3 vs stack3 + hotspot branch + holo-embedding auxiliary head\n"
i = next(i for i in range(len(cells)) if "ARMS = [(" in src(i))
cells[i]["source"] = r'''
KEEP_SIDE = list(range(TOPO_DIM)); m_ = torch.zeros(SIDE_DIM); m_[KEEP_SIDE] = 1.0
for d in dataset: d.x[:, N_PRE:] = d.x[:, N_PRE:] * m_
AUX_W = 0.3

def val_pr(model):
    ps, ys = [], []
    with torch.no_grad():
        for b in DataLoader(splits["val"], batch_size=16):
            b = b.to(DEVICE); lg, _ = model(b); ps.append(torch.sigmoid(lg).cpu().numpy()); ys.append(b.y.cpu().numpy())
    return average_precision_score(np.concatenate(ys), np.concatenate(ps))

def fit_es(seed, pretrained_path, flags, epochs=40, patience=8, lam=0.1, batch=16):
    """Phase 129/291/297/299 fit semantics; keep the best val-PR state, stop after 8 bad epochs."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = LocalizingEGNN(N_PRE, SIDE_DIM, use_loc_head=True, plm_dim=1280, plm2_dim=1024, plm4_dim=1280, **flags).to(DEVICE)
    missing, unexpected = model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE), strict=False)
    assert not [k for k in missing if not k.startswith(("plm_proj", "plm2_proj", "plm4_proj", "hemb_head", "hot_proj"))] and not unexpected
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

ARMS = [("stack3", {}), ("stack3_hot_studemb", dict(hot=True, auxemb=True))]
T0 = time.time(); ALL = {}
for name, flags in ARMS:
    ALL[name] = {"seeds": {}}
    for seed in range(8):
        model, v, ep = fit_es(seed, locate(f"pretrain266_full_seed{seed}.pt"), flags)
        ALL[name]["seeds"][seed] = {"val": predict_split(model, "val"), "test": predict_split(model, "test"), "val_pr": float(v), "epochs": ep}
        torch.save(ALL, WORK / "phase305_test_preds.pt")
        print(f"  [{name}] seed {seed}: val PR {v:.4f} ({ep} ep)  [{(time.time()-T0)/60:.1f}m]", flush=True)
print("done")
'''
d = ROOT_OUT / SLUG; d.mkdir(parents=True, exist_ok=True)
o = d / "phase305_stack3hs_test.ipynb"; o.write_text(json.dumps(nb, indent=1) + "\n")
meta = json.load(open(tmp / "303" / "kernel-metadata.json")); meta.update(id="blackpearls1/phase305-stack3hs-test", title="Phase305 Stack3hs Test", code_file=o.name)
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n"); print("wrote", o)
