#!/usr/bin/env python3
"""Phase 280a: homology check for 2,300 unused, already-labelled apo/holo pairs (the "preformed" and "neither" entries of
pmfran4252-afk/cryptic-site-benchmark's cryptic_benchmark.json) that we want to add as EXTRA CryptoBench-style training data.
Must not share a >=30%-identity homolog with anything we have ever reported an evaluation number on: CryptoBench's own
official val (60) + test (182) chains, PocketMiner's test set (35, Phase 277a), or this same repository's own 99
"cryptic" entries (Phase 277b) -- all of those are eval sets this project's results already rest on. CPU kernel,
mmseqs2, identical settings to every earlier leak check in this project (30% identity, 80% coverage, cov-mode 1)."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
cand_json = (HERE / "candidate_entries.json").read_text()
cryptic_json = (HERE / "cryptic99_entries.json").read_text()
C = []
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})
code("!pip install -q torch-geometric==2.6.1 biopython")
code("CANDIDATE_JSON = " + repr(cand_json) + "  # the 2,302 preformed+neither entries, embedded verbatim (no runtime fetch of the source file needed)")
code("CRYPTIC99_JSON = " + repr(cryptic_json) + "  # this repository's own 99 cryptic entries (Phase 277b), embedded so this kernel needs no local file lookup")
code(r'''
import gzip, json, os, shutil, subprocess, time, urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch

def locate(f):
    c = []
    for r in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if r.exists(): c.extend(r.rglob(f))
    if not c: raise FileNotFoundError(f)
    return max(c, key=lambda p: p.stat().st_size)

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA3_TO_1={"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
          "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
          "THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M"}
RAW = Path("/tmp/raw280"); RAW.mkdir(parents=True, exist_ok=True)

# --- protected set 1: CryptoBench's own official val + test sequences (decoded losslessly from the one-hot block) ---
cb = torch.load(locate("phase48_cryptobench_graphs_esm3b.pt"), map_location="cpu", weights_only=False)["dataset"]
def cb_seq(d):
    idx = d.x[:, :21].float().argmax(1); assert bool((d.x[:, :21].float().sum(1) == 1).all())
    return "".join(AA_ALPHABET[i] for i in idx.tolist())
cb_valtest = {(str(d.pdb_id).lower(), str(d.chain_id)): cb_seq(d) for d in cb if str(d.split) in ("val", "test")}
del cb
print("CryptoBench val+test chains:", len(cb_valtest))
''')
code(r'''
# --- protected set 2: PocketMiner's 35-protein test set (same fetch as Phase 277a/277c) ---
BASE_PM = "https://raw.githubusercontent.com/Mickdub/gvp/pocket_pred/data/pm-dataset"
def fetch_bytes(url, tries=4):
    for a in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r: return r.read()
        except Exception: time.sleep(2 * (a + 1))
    raise RuntimeError(f"failed: {url}")

WORK = Path("/kaggle/working"); (WORK).mkdir(parents=True, exist_ok=True)
(WORK / "test_ids.npy").write_bytes(fetch_bytes(f"{BASE_PM}/test_apo_ids_with_chainids.npy"))
test_ids = [str(x) for x in np.load(WORK / "test_ids.npy")]
import urllib.request as ur
api = json.loads(ur.urlopen("https://api.github.com/repos/Mickdub/gvp/contents/data/pm-dataset/apo-structures?ref=pocket_pred", timeout=30).read())
pm_files = [f["name"] for f in api]

def match_file(idc):
    pid, ch = idc[:4], idc[4:]
    cands = [f for f in pm_files if f.lower().startswith(pid.lower())]
    exact = [f for f in cands if f[:len(pid) + len(ch)].lower() == (pid + ch).lower()]
    return exact[0] if exact else (cands[0] if len(cands) == 1 else None)

from Bio.PDB import PDBParser
def seq_of_pm(idc):
    fn = match_file(idc); dst = RAW / fn
    if not dst.exists(): dst.write_bytes(fetch_bytes(f"{BASE_PM}/apo-structures/{fn}"))
    model = next(PDBParser(QUIET=True, PERMISSIVE=True).get_structure("x", str(dst)).get_models())
    chains = list(model); ch_hint = idc[4:]
    chain = next((c for c in chains if c.id == ch_hint), None) or max(chains, key=lambda c: sum(1 for r in c if r.get_resname().strip().upper() in AA3_TO_1 and "CA" in r))
    residues = [r for r in chain if r.get_resname().strip().upper() in AA3_TO_1 and "CA" in r]
    return "".join(AA3_TO_1.get(r.get_resname().strip().upper(), "X") for r in residues)

pm_test = {idc: seq_of_pm(idc) for idc in test_ids}
print("PocketMiner test sequences:", len(pm_test))
''')
code(r'''
# --- protected set 3: this repository's own 99 "cryptic" entries (Phase 277b), fetched fresh from RCSB ---
# --- candidate set: the 2,302 (2,300 unique) "preformed"/"neither" entries we want to add as training data ---
def fetch_rcsb(pid):
    for ext in ("cif", "pdb"):
        p = RAW / f"{pid}.{ext}.gz"
        if p.exists() and p.stat().st_size > 500: return pid, p
        try:
            with urllib.request.urlopen(f"https://files.rcsb.org/download/{pid}.{ext}.gz", timeout=60) as r: b = r.read()
            if len(b) > 500: p.write_bytes(b); return pid, p
        except Exception: continue
    return pid, None

def seq_of_rcsb(pid, chain_id):
    _, path = fetch_rcsb(pid)
    if path is None: return None
    from Bio.PDB import MMCIFParser
    parser = MMCIFParser(QUIET=True) if ".cif" in path.name else PDBParser(QUIET=True, PERMISSIVE=True)
    try:
        with gzip.open(path, "rt") as fh: model = next(parser.get_structure(pid, fh).get_models())
        ch = model[chain_id]
    except Exception: return None
    residues = [r for r in ch if r.get_resname().strip().upper() in AA3_TO_1 and "CA" in r]
    if not residues: return None
    return "".join(AA3_TO_1.get(r.get_resname().strip().upper(), "X") for r in residues)

cryptic_entries = json.loads(CRYPTIC99_JSON)
candidates = json.loads(CANDIDATE_JSON)
need = sorted({e["apo"] for e in cryptic_entries} | {e["apo"] for e in candidates})
with ThreadPoolExecutor(16) as ex: list(ex.map(fetch_rcsb, need))

csb_cryptic = {}
for e in cryptic_entries:
    s = seq_of_rcsb(e["apo"], e["apo_chain"])
    if s: csb_cryptic[f'{e["apo"]}_{e["apo_chain"]}'] = s
print("cryptic-site-benchmark cryptic-99 sequences:", len(csb_cryptic))

seen = set(); cand_seq = {}
for e in candidates:
    key = f'{e["apo"]}_{e["apo_chain"]}'
    if key in seen: continue
    seen.add(key)
    s = seq_of_rcsb(e["apo"], e["apo_chain"])
    if s: cand_seq[key] = s
print("candidate (preformed/neither) sequences:", len(cand_seq), "of", len(seen), "unique keys")
''')
code(r'''
subprocess.run("curl -sL https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -o /tmp/mm.tar.gz && tar xzf /tmp/mm.tar.gz -C /tmp", shell=True, check=True, timeout=600)
mm = "/tmp/mmseqs/bin/mmseqs"; os.chmod(mm, 0o755)
W = Path("/tmp/hom280"); shutil.rmtree(W, ignore_errors=True); W.mkdir()
lines = []
lines += [f">CBVALTEST__{k[0]}_{k[1]}\n{s}\n" for k, s in cb_valtest.items()]
lines += [f">PM__{k}\n{s}\n" for k, s in pm_test.items()]
lines += [f">CSBCRYPTIC__{k}\n{s}\n" for k, s in csb_cryptic.items()]
lines += [f">CAND__{k}\n{s}\n" for k, s in cand_seq.items()]
(W / "all.fasta").write_text("".join(lines))
subprocess.run([mm, "easy-cluster", str(W / "all.fasta"), str(W / "res"), str(W / "tmp"),
                "--min-seq-id", "0.30", "-c", "0.8", "--cov-mode", "1", "-v", "1"], check=True, capture_output=True, timeout=3600)
rep = defaultdict(list)
for line in (W / "res_cluster.tsv").read_text().splitlines():
    a, b = line.split("\t"); rep[a].append(b)
for r_, members in rep.items():
    if r_ not in members: members.append(r_)
protected_prefixes = ("CBVALTEST__", "PM__", "CSBCRYPTIC__")
leaked = set()
for members in rep.values():
    if any(m.startswith(protected_prefixes) for m in members):
        leaked.update(m for m in members if m.startswith("CAND__"))
clean_keys = [k for k in cand_seq if f"CAND__{k}" not in leaked]
print(f"candidates: {len(cand_seq)} total; {len(leaked)} leaked (homologous to a protected eval set); {len(clean_keys)} clean and usable as new training data")
json.dump({"clean_keys": clean_keys, "leaked_keys": [k for k in cand_seq if f"CAND__{k}" in leaked]},
          open("/kaggle/working/phase280a_clean_candidates.json", "w"), indent=1)
print("saved -> phase280a_clean_candidates.json")
''')
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
d = HERE / "280a"
(d / "phase280a_homology.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
(d / "kernel-metadata.json").write_text(json.dumps({"id": "blackpearls1/phase280a-extra-data-homology", "title": "Phase280a Extra Data Homology",
    "code_file": "phase280a_homology.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": False, "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
    "kernel_sources": ["blackpearls1/tda-egnn-phase48-cryptobench-esm3b"]}, indent=2) + "\n")
print("wrote", d)
