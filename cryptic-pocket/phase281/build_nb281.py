#!/usr/bin/env python3
"""Phase 281: a real MD pilot -- does actual physics-based dynamics (not the elastic-network approximation that already failed, Phase 270-272)
show more local flexibility/opening at the residues that are TRULY cryptic-pocket-lining than elsewhere in the same protein?

17 CryptoBench test proteins with a strict, independently-verified "genuinely cryptic" label (Phase 276's crypticity audit) -- apo structure
only (no ligand, matching the real task), stripped to the single target chain, solvated, minimized, 200 ps equilibration, 5 ns unbiased
production MD (GPU, CUDA via a self-contained micromamba+conda-forge openmm install -- Phase 278 probes established this is the only install
path that actually finds the GPU: pip's wheel has no CUDA platform, and this Kaggle image's "mamba" is an unrelated tool of the same name).

Per residue: CA-RMSF over the production trajectory (aligned to the average structure), and the mean/std/max change in heavy-atom neighbour
count within 10 A of its CA relative to the starting (minimized) structure, sampled every 50 ps -- the same "does this residue's neighbourhood
loosen or tighten" idea the elastic-network approximation tested and failed, now from real sampled dynamics instead of a linear approximation.
Saved per-protein so the correlation against the true site-residue labels can be checked before any decision to build a full retrain on it."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
entries_json = (HERE / "md_pilot_entries.json").read_text()
C = []
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

code("!curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /tmp bin/micromamba\n!chmod +x /tmp/bin/micromamba\n!/tmp/bin/micromamba --version")
code("!/tmp/bin/micromamba create -y -q -p /tmp/mm_env -c conda-forge -c nvidia python=3.11 openmm cudatoolkit=11.8 pdbfixer numpy scipy 2>&1 | tail -20")

driver = r'''
import gzip, json, time, urllib.request
from pathlib import Path
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from pdbfixer import PDBFixer
from scipy.spatial import cKDTree

ENTRIES = json.loads(open("/kaggle/working/md_pilot_entries.json").read())
plat = mm.Platform.getPlatformByName("CUDA")
print(f"platform: {plat.getName()}  |  {len(ENTRIES)} proteins queued")

WORK = Path("/kaggle/working"); RAW = Path("/tmp/raw281"); RAW.mkdir(parents=True, exist_ok=True)
NBR_R = 10.0            # angstrom, same radius used throughout this project's geometry/movement features
SAVE_EVERY_PS = 50.0    # heavy-atom snapshots for the neighbour-count feature
DT_PS = 0.002
EQUIL_PS, PROD_PS = 200.0, 5000.0

def fetch(pid):
    p = RAW / f"{pid}.pdb"
    if p.exists() and p.stat().st_size > 500: return p
    urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pid.upper()}.pdb", str(p))
    return p

def run_one(pid, chain_id, site_resnums):
    t0 = time.time()
    path = fetch(pid)
    fixer = PDBFixer(filename=str(path))
    keep = chain_id
    all_ids = [c.id for c in fixer.topology.chains()]
    fixer.removeChains(chainIds=[c for c in all_ids if c != keep])
    fixer.findMissingResidues(); fixer.missingResidues = {}          # do NOT model in missing loops -- real crystallised atoms only
    fixer.findNonstandardResidues(); fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms(); fixer.addMissingAtoms(); fixer.addMissingHydrogens(7.0)

    residues = list(fixer.topology.residues())
    resnum_of = {}                                                   # topology residue index -> author resnum string (matches CryptoBench convention)
    for i, r in enumerate(residues):
        resnum_of[i] = str(r.id)
    n_res = len(residues)
    site_idx = {i for i, rn in resnum_of.items() if rn in set(site_resnums)}
    if len(site_idx) < 0.5 * len(site_resnums):
        return None, f"only {len(site_idx)}/{len(site_resnums)} site residues found post-PDBFixer"

    modeller = app.Modeller(fixer.topology, fixer.positions)
    ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    modeller.addSolvent(ff, model="tip3p", padding=1.0 * unit.nanometer, ionicStrength=0.15 * unit.molar)
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer, constraints=app.HBonds)
    integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, DT_PS * unit.picoseconds)
    sim = app.Simulation(modeller.topology, system, integrator, plat)
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy(maxIterations=1000)

    protein_atom_idx = [a.index for a in modeller.topology.atoms() if a.residue.name not in ("HOH", "NA", "CL")]
    ca_idx = [a.index for a in modeller.topology.atoms() if a.name == "CA" and a.residue.name not in ("HOH", "NA", "CL")]
    assert len(ca_idx) == n_res, (len(ca_idx), n_res)

    sim.context.setVelocitiesToTemperature(300 * unit.kelvin)
    sim.step(int(EQUIL_PS / DT_PS))

    start_state = sim.context.getState(getPositions=True)
    start_heavy = np.array(start_state.getPositions(asNumpy=True).value_in_unit(unit.angstrom))[protein_atom_idx]
    start_ca = np.array(start_state.getPositions(asNumpy=True).value_in_unit(unit.angstrom))[ca_idx]
    start_tree = cKDTree(start_heavy)
    start_nbr = np.array([len(start_tree.query_ball_point(c, NBR_R)) for c in start_ca])

    n_snap = int(PROD_PS / SAVE_EVERY_PS)
    steps_per_snap = int(SAVE_EVERY_PS / DT_PS)
    ca_traj = np.zeros((n_snap, len(ca_idx), 3))
    nbr_traj = np.zeros((n_snap, len(ca_idx)))
    for i in range(n_snap):
        sim.step(steps_per_snap)
        st = sim.context.getState(getPositions=True)
        pos = np.array(st.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
        heavy = pos[protein_atom_idx]; ca = pos[ca_idx]
        ca_traj[i] = ca
        tree = cKDTree(heavy)
        nbr_traj[i] = [len(tree.query_ball_point(c, NBR_R)) for c in ca]

    def kabsch_fit(mobile, ref):
        mobile_c = mobile - mobile.mean(0); ref_c = ref - ref.mean(0)
        U, S, Vt = np.linalg.svd(mobile_c.T @ ref_c)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, d]) @ U.T
        return (R @ mobile_c.T).T + ref.mean(0)

    ref = ca_traj[0].copy()                                          # rigid-body alignment (Kabsch) to remove whole-molecule rotation/translation,
    for _ in range(3):                                                # not just internal flexibility -- iterate a few times toward the mean structure
        aligned = np.stack([kabsch_fit(f, ref) for f in ca_traj])
        ref = aligned.mean(0)
    ca_traj = aligned
    ca_mean = ca_traj.mean(0)
    rmsf = np.sqrt(((ca_traj - ca_mean) ** 2).sum(-1).mean(0))       # per-residue CA-RMSF, angstrom, after removing rigid-body motion
    nbr_delta = nbr_traj - start_nbr[None, :]                        # positive = MORE neighbours over time (tightening); negative = opening
    out = dict(pdb=pid, chain=chain_id, n_res=n_res, site_idx=sorted(site_idx),
               rmsf=rmsf.tolist(), nbr_mean_delta=nbr_delta.mean(0).tolist(), nbr_std=nbr_traj.std(0).tolist(),
               nbr_min_delta=nbr_delta.min(0).tolist(), nbr_max_delta=nbr_delta.max(0).tolist(),
               resnum_of=[resnum_of[i] for i in range(n_res)], wall_s=time.time() - t0)
    return out, "ok"

results = {}
for e in ENTRIES:
    key = f"{e['apo']}_{e['apo_chain']}"
    try:
        out, msg = run_one(e["apo"], e["apo_chain"], e["site_residues"])
    except Exception as ex:
        out, msg = None, f"{type(ex).__name__}: {ex}"
    print(f"  {key}: {msg}" + (f"  [{out['wall_s']/60:.1f} min]" if out else ""), flush=True)
    if out: results[key] = out
    json.dump(results, open(WORK / "phase281_md_pilot_results.json", "w"))     # incremental save
print(f"done: {len(results)}/{len(ENTRIES)} proteins succeeded")
'''
(Path("/tmp") / "_unused").write_text("")  # no-op, keeps this cell purely a string builder above
code("(open('/kaggle/working/md_pilot_entries.json','w')).write(" + repr(entries_json) + ")")
code("(open('/tmp/mm_driver.py','w')).write(" + repr(driver) + ")\nimport subprocess, sys\nr = subprocess.run('/tmp/bin/micromamba run -p /tmp/mm_env python -u /tmp/mm_driver.py', shell=True, capture_output=True, text=True)\nprint(r.stdout[-20000:]); print('STDERR TAIL:', r.stderr[-3000:])")
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
d = HERE / "run"; d.mkdir(exist_ok=True)
(d / "phase281_md_pilot.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
(d / "kernel-metadata.json").write_text(json.dumps({"id": "blackpearls1/phase281-md-pilot", "title": "Phase281 Md Pilot", "code_file": "phase281_md_pilot.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4",
    "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": []}, indent=2) + "\n")
print("wrote", d)
