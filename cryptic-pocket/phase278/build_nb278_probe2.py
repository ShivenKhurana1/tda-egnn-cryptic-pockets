#!/usr/bin/env python3
"""Phase 278 probe #2: the first attempt (pip install openmm) registered only Reference/CPU/OpenCL platforms -- no CUDA -- giving 0.17 ns/hour,
118 hours for a 20 ns run. That is infeasible for any pilot. This retries via conda-forge's openmm+cudatoolkit build, which ships a CUDA
platform plugin matched to a real cudatoolkit, and checks nvidia-smi / driver visibility first to rule out "no GPU attached" as the cause."""
import json
from pathlib import Path
C = []
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})
code("!nvidia-smi")
code("!which conda mamba 2>&1; conda --version 2>&1")
code("!mamba install -y -q -c conda-forge openmm cudatoolkit=11.8 2>&1 | tail -40")
code(r'''
import openmm as mm
print("OpenMM version:", mm.Platform.getOpenMMVersion())
for i in range(mm.Platform.getNumPlatforms()):
    print(" platform", i, mm.Platform.getPlatform(i).getName())
''')
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
d = Path("probe2"); d.mkdir(exist_ok=True)
(d / "phase278_md_probe2.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
(d / "kernel-metadata.json").write_text(json.dumps({"id": "blackpearls1/phase278-md-probe2", "title": "Phase278 Md Probe2", "code_file": "phase278_md_probe2.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4",
    "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": []}, indent=2) + "\n")
print("wrote", d)
