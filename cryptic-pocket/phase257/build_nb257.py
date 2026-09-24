#!/usr/bin/env python3
"""Phase 257: the remaining folds (4-9) of the Phase 255 classification geometry control, on Kaggle CPU.

One kernel = one (fold, arm) pair. Its two seeds run as two SEPARATE PROCESSES (2 threads each): a timing probe showed one training
uses only ~2 of the 4 cores (2 threads 502 s/epoch, 4 threads 539 s/epoch), so two processes use the whole machine. Separate processes
(not fork) avoid the OpenMP-after-fork hazard. Identical to Phase 255 in data, folds, model, loss, early stopping and seeds.

usage: build_nb257.py <fold 4-9> <arm no-topo|full|geom|geom+topo> <output_root>"""
import ast, builtins, json, sys
from pathlib import Path

FOLD = int(sys.argv[1]); ARM = sys.argv[2]; ROOT_OUT = Path(sys.argv[3]).resolve()
assert FOLD in range(4, 10)
NAME = f"f{FOLD}{ {'no-topo': 'nt', 'full': 'fu', 'geom': 'ge', 'geom+topo': 'gt'}[ARM] }"
HERE = Path(__file__).resolve().parent
nb = json.load(open(HERE.parent / "phase255" / "a" / "phase255a_cls_geometry_scale3.ipynb"))
cells = nb["cells"]; src = lambda i: "".join(cells[i]["source"])

WORKER = src(4) + "\n\n" + src(6) + f'''

# ---- one seed of one (fold, arm), run as its own process ----------------------------------------------------------
import sys, resource
SEED = int(sys.argv[1]); FOLD_TO_RUN, ARM_TO_RUN = {FOLD}, "{ARM}"
torch.set_num_threads(2)
print("worker seed", SEED, "| device", DEVICE, "| threads", torch.get_num_threads(), "| RSS MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, flush=True)
''' + r'''
N_FOLDS, EPOCHS, PATIENCE, BATCH = 10, 40, 8, 16
KEEP_SIDE = {"no-topo": [], "full": list(range(TOPO_DIM)), "geom": list(range(TOPO_DIM, SIDE_DIM)), "geom+topo": list(range(SIDE_DIM))}
m_ = torch.zeros(SIDE_DIM); m_[KEEP_SIDE[ARM_TO_RUN]] = 1.0
for d in dataset: d.x[:, N_PRE:] = d.x[:, N_PRE:] * m_          # in place: unselected side columns zeroed (identical result to Phase 255's apply_arm)
folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=1337).split(np.arange(len(dataset))))
tr_idx, te_idx = folds[FOLD_TO_RUN]
t0 = time.time()
rng = np.random.RandomState(500 + SEED)
perm = rng.permutation(tr_idx)
nv = max(1, int(0.15 * len(perm)))
va_ds = [dataset[i] for i in perm[:nv]]; tr_ds = [dataset[i] for i in perm[nv:]]; te_ds = [dataset[i] for i in te_idx]
torch.manual_seed(SEED); np.random.seed(SEED)
m = Net(N_PRE, SIDE_DIM).to(DEVICE)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
tl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True); vl = DataLoader(va_ds, batch_size=BATCH)
best, state, bad = -1, None, 0
for ep in range(EPOCHS):
    run_epoch(m, tl, opt); v = run_epoch(m, vl); sch.step()
    print(f"    seed {SEED} epoch {ep}: val PR {v['pr']:.3f} [{(time.time()-t0)/60:.0f}m]", flush=True)
    if v["pr"] > best: best, state, bad = v["pr"], {k: t.clone() for k, t in m.state_dict().items()}, 0
    else:
        bad += 1
        if bad >= PATIENCE: break
m.load_state_dict(state)
te = run_epoch(m, DataLoader(te_ds, batch_size=BATCH))
rec = {"arm": ARM_TO_RUN, "fold": FOLD_TO_RUN, "seed": SEED, "val_pr": float(best), "test_pr": float(te["pr"]),
       "test_roc": float(te["roc"]), "n_test": int(len(te["y"])), "n_pos": int(te["y"].sum()), "epochs": ep + 1}
json.dump(rec, open(f"/kaggle/working/rec_seed{SEED}.json", "w"), indent=2, default=float)
print(f"DONE seed {SEED}: test PR {te['pr']:.3f} ROC {te['roc']:.3f} [{(time.time()-t0)/60:.0f}m] peak RSS MB {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}", flush=True)
'''

# static check of the worker: no undefined names (the failure mode that cost hours before)
tree = ast.parse(WORKER); defined = set(dir(builtins)); bad = []
for n in ast.walk(tree):
    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store): defined.add(n.id)
    elif isinstance(n, (ast.FunctionDef, ast.ClassDef)): defined.add(n.name)
    elif isinstance(n, ast.arg): defined.add(n.arg)
    elif isinstance(n, (ast.Import, ast.ImportFrom)):
        for a in n.names: defined.add((a.asname or a.name).split(".")[0])
    elif isinstance(n, ast.ExceptHandler) and n.name: defined.add(n.name)
for n in ast.walk(tree):
    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in defined: bad.append(n.id)
assert not bad, f"worker has undefined names: {sorted(set(bad))}"

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})
md(f'''
# Phase 257 {NAME} — classification geometry control at the 7,513-chain pool, CPU: fold {FOLD}, arm `{ARM}`, seeds 0 and 1

Continuation of Phase 255 (folds 0-3 finished on GPU; Kaggle's 30 h weekly GPU quota then ran out). Same data, folds (KFold seed 1337), model,
loss, early stopping and seeds as Phase 255; only the device differs. The two seeds run as two separate 2-thread processes so both use the
4 cores. Primary comparison across the ten folds: `geom+topo` vs `geom` on test PR-AUC.
''')
code("!pip install -q torch-geometric==2.6.1")
code(f"WORKER_SRC = {WORKER!r}\nopen('/kaggle/working/worker.py', 'w').write(WORKER_SRC)\nprint('worker script written,', len(WORKER_SRC), 'chars')")
code(r'''
import subprocess, sys, os, time, json
procs = []
for seed in (0, 1):
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    log = open(f"/kaggle/working/worker_seed{seed}.log", "w")
    procs.append((seed, subprocess.Popen([sys.executable, "-u", "/kaggle/working/worker.py", str(seed)], stdout=log, stderr=subprocess.STDOUT, env=env), log))
t0 = time.time()
while any(p.poll() is None for _, p, _ in procs):
    time.sleep(600)
    for seed, p, _ in procs:
        lines = open(f"/kaggle/working/worker_seed{seed}.log").read().strip().split("\n")
        print(f"[{(time.time()-t0)/60:.0f}m] seed {seed} (rc={p.poll()}): {lines[-1][:160]}", flush=True)
for seed, p, log in procs:
    log.close(); print(f"--- seed {seed} exit code {p.returncode}; last log lines:"); print("\n".join(open(f"/kaggle/working/worker_seed{seed}.log").read().strip().split("\n")[-4:]))
assert all(p.returncode == 0 for _, p, _ in procs), "a worker failed -- see its log above"
recs = [json.load(open(f"/kaggle/working/rec_seed{s}.json")) for s in (0, 1)]
json.dump({"records": recs, "folds": [recs[0]["fold"]]}, open("/kaggle/working/phase257___NAME___records.json", "w"), indent=2)
print("saved phase257___NAME___records.json:", [(r["seed"], round(r["test_pr"], 4)) for r in recs])
'''.replace("__NAME__", NAME))
out = {"cells": C, "metadata": nb["metadata"], "nbformat": 4, "nbformat_minor": 5}
d = ROOT_OUT / NAME; d.mkdir(parents=True, exist_ok=True)
o = d / f"phase257{NAME}_cls_cpu.ipynb"; o.write_text(json.dumps(out, indent=1) + "\n")
meta = {"id": f"blackpearls1/phase257-{NAME}-cls-cpu", "title": f"Phase257 {NAME} Cls Cpu", "code_file": o.name, "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["blackpearls1/tda-egnn-phase52-esm-dihedral-scale3", "blackpearls1/phase250a-geometry-scale3",
                           "blackpearls1/phase250b-geometry-scale3", "blackpearls1/phase250c-geometry-scale3"]}
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote", o)
