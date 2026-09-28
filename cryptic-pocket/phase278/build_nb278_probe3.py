#!/usr/bin/env python3
"""Phase 278 probe #3: pip's openmm wheel has no CUDA platform (probe #1: Reference/CPU/OpenCL only), and this Kaggle image has no real
conda/mamba (probe #2's "mamba" is an unrelated Python test framework of the same name). Final attempt: bootstrap a real, self-contained
micromamba (downloaded static binary, no pre-existing package manager needed) and install openmm+cuda through it -- the standard way to get
GPU-accelerated OpenMM on a bare Linux box. If this also fails to produce a CUDA platform, GPU-accelerated MD is not practical on Kaggle
within reasonable effort, and this line of investigation should be closed out rather than pursued further."""
import json
from pathlib import Path
C = []
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})
code(r'''
!curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /tmp bin/micromamba
!chmod +x /tmp/bin/micromamba
!/tmp/bin/micromamba --version
''')
code(r'''
import subprocess, time
t0 = time.time()
r = subprocess.run(
    "/tmp/bin/micromamba create -y -p /tmp/mm_env -c conda-forge -c nvidia "
    "python=3.11 openmm cudatoolkit=11.8 pdbfixer 2>&1 | tail -60",
    shell=True, capture_output=True, text=True
)
print(r.stdout[-4000:])
print(f"env creation took {(time.time()-t0)/60:.1f} min")
''')
code(r'''
import subprocess
r = subprocess.run(
    "/tmp/bin/micromamba run -p /tmp/mm_env python -c \""
    "import openmm as mm; print('OpenMM version:', mm.Platform.getOpenMMVersion());"
    "[print(' platform', i, mm.Platform.getPlatform(i).getName()) for i in range(mm.Platform.getNumPlatforms())]\"",
    shell=True, capture_output=True, text=True
)
print("STDOUT:", r.stdout)
print("STDERR:", r.stderr[-1500:])
''')
code(r'''
# if a CUDA platform exists, do the same timed probe as before (1UBQ, solvate, minimize, equilibrate, time production) via micromamba's python
script = r"""
import time, urllib.request
import openmm as mm, openmm.app as app, openmm.unit as unit
from pdbfixer import PDBFixer
plat = mm.Platform.getPlatformByName('CUDA')
urllib.request.urlretrieve('https://files.rcsb.org/download/1UBQ.pdb', '/tmp/in.pdb')
fixer = PDBFixer(filename='/tmp/in.pdb')
fixer.findMissingResidues(); fixer.findMissingAtoms(); fixer.addMissingAtoms(); fixer.addMissingHydrogens(7.0)
modeller = app.Modeller(fixer.topology, fixer.positions)
ff = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
modeller.addSolvent(ff, model='tip3p', padding=1.0*unit.nanometer, ionicStrength=0.15*unit.molar)
system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometer, constraints=app.HBonds)
integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
sim = app.Simulation(modeller.topology, system, integrator, plat)
sim.context.setPositions(modeller.positions)
sim.minimizeEnergy(maxIterations=500)
sim.context.setVelocitiesToTemperature(300*unit.kelvin)
sim.step(5000)
t0 = time.time(); sim.step(25000); el = time.time() - t0
ns_per_hour = (25000*0.002/1000)/(el/3600)
print(f'production 50 ps in {el:.1f}s -> {ns_per_hour:.2f} ns/hour on CUDA')
print(f'estimated 20 ns run: {20/ns_per_hour:.2f} hours')
"""
open("/tmp/timed_probe.py", "w").write(script)
import subprocess
r = subprocess.run("/tmp/bin/micromamba run -p /tmp/mm_env python /tmp/timed_probe.py", shell=True, capture_output=True, text=True)
print("STDOUT:", r.stdout[-2000:])
print("STDERR:", r.stderr[-1500:])
''')
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
d = Path("probe3"); d.mkdir(exist_ok=True)
(d / "phase278_md_probe3.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
(d / "kernel-metadata.json").write_text(json.dumps({"id": "blackpearls1/phase278-md-probe3", "title": "Phase278 Md Probe3", "code_file": "phase278_md_probe3.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4",
    "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": []}, indent=2) + "\n")
print("wrote", d)
