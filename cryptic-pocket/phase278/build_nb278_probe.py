#!/usr/bin/env python3
"""Phase 278 probe: can we run real, GPU-accelerated molecular dynamics on Kaggle at all, and how fast?
Installs OpenMM, fixes/preps one small apo structure (PDBFixer: add missing atoms/hydrogens), solvates it,
minimizes, equilibrates briefly, and runs a short production MD run -- timed, so we know ns/hour on a T4
before designing any real pilot. Nothing here is used for a result; it's a feasibility and timing check."""
import json
from pathlib import Path
C = []
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})
code("!pip install -q openmm pdbfixer 2>&1 | tail -5")
code(r'''
import time, urllib.request
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from pdbfixer import PDBFixer

print("OpenMM version:", mm.Platform.getOpenMMVersion())
for i in range(mm.Platform.getNumPlatforms()):
    print(" platform", i, mm.Platform.getPlatform(i).getName())
plat = mm.Platform.getPlatformByName("CUDA") if any(mm.Platform.getPlatform(i).getName()=="CUDA" for i in range(mm.Platform.getNumPlatforms())) else mm.Platform.getPlatformByName("CPU")
print("using platform:", plat.getName())
''')
code(r'''
# small, well-behaved apo test protein (chosen arbitrarily -- just for a timing/feasibility probe)
pdb_id = "1UBQ"  # ubiquitin, 76 residues, no cofactors, classic MD test case
urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.pdb", "/kaggle/working/in.pdb")

t0 = time.time()
fixer = PDBFixer(filename="/kaggle/working/in.pdb")
fixer.findMissingResidues()
fixer.findMissingAtoms()
fixer.addMissingAtoms()
fixer.addMissingHydrogens(7.0)
print(f"PDBFixer done in {time.time()-t0:.1f}s; atoms: {fixer.topology.getNumAtoms()}")

modeller = app.Modeller(fixer.topology, fixer.positions)
forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
t0 = time.time()
modeller.addSolvent(forcefield, model="tip3p", padding=1.0*unit.nanometer, ionicStrength=0.15*unit.molar)
print(f"solvation done in {time.time()-t0:.1f}s; atoms with water: {modeller.topology.getNumAtoms()}")

system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometer,
                                  constraints=app.HBonds)
integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
sim = app.Simulation(modeller.topology, system, integrator, plat)
sim.context.setPositions(modeller.positions)

t0 = time.time()
sim.minimizeEnergy(maxIterations=500)
print(f"minimization done in {time.time()-t0:.1f}s")

sim.context.setVelocitiesToTemperature(300*unit.kelvin)
t0 = time.time()
EQUIL_STEPS = 5000   # 10 ps
sim.step(EQUIL_STEPS)
t_equil = time.time() - t0
print(f"equilibration: {EQUIL_STEPS} steps (10 ps) in {t_equil:.1f}s")

t0 = time.time()
PROD_STEPS = 25000   # 50 ps production, just for a rate measurement
sim.step(PROD_STEPS)
t_prod = time.time() - t0
ns_per_hour = (PROD_STEPS * 0.002 / 1000) / (t_prod / 3600)
print(f"production: {PROD_STEPS} steps (50 ps) in {t_prod:.1f}s  ->  {ns_per_hour:.2f} ns/hour on this platform/system size")
print(f"estimated time for a 20 ns run at this rate: {20/ns_per_hour:.2f} hours")
''')
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
d = Path("probe"); d.mkdir(exist_ok=True)
(d / "phase278_md_probe.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
(d / "kernel-metadata.json").write_text(json.dumps({"id": "blackpearls1/phase278-md-probe", "title": "Phase278 Md Probe", "code_file": "phase278_md_probe.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4",
    "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": []}, indent=2) + "\n")
print("wrote", d)
