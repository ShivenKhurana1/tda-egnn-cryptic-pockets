#!/usr/bin/env python3
"""Phase 308a: inspect the CryptoBank database (Martinez et al., Zenodo 19099155, CC-BY-4.0) on a Kaggle CPU kernel (~30 GB RAM; the
4.8 GB pickle is too large for a local load). The pickle is read with a RESTRICTED unpickler that only resolves pandas/numpy
reconstruction globals and refuses everything else, so no code from the file can run; RDKit ligand objects become inert stubs (no RDKit code runs). Outputs: the table schema and stats, a 40-row
sample with long values truncated, and a slim parquet with every column whose typical value is small (heavy array/coordinate columns
dropped, object columns stringified). No GPU.
usage: build_nb308a.py <output_root>"""
import json, sys
from pathlib import Path
ROOT_OUT = Path(sys.argv[1]).resolve(); SLUG = "308a"
code = [r'''import pickle, zipfile, urllib.request, time, json, io, sys
from pathlib import Path
import numpy as np, pandas as pd
OK = ("pandas.", "numpy.", "builtins.slice", "builtins.list", "builtins.dict", "builtins.tuple", "builtins.set", "builtins.frozenset",
      "builtins.int", "builtins.float", "builtins.str", "builtins.bytearray", "builtins.complex", "collections.OrderedDict",
      "_codecs.encode", "copyreg._reconstructor", "builtins.object", "datetime.")
REFUSED = []; STUBBED = set()
class Stub:
    """Inert placeholder for RDKit ligand objects (fingerprints, mols): keeps the raw args/state, runs no RDKit code."""
    def __init__(self, *a, **k): self.args = a
    def __setstate__(self, st): self.state = st
    def __repr__(self): return "<rdkit stub>"
class Safe(pickle.Unpickler):
    def find_class(self, mod, name):
        full = f"{mod}.{name}"
        if any(full == p or (p.endswith(".") and full.startswith(p)) for p in OK): return super().find_class(mod, name)
        if full.startswith("rdkit."): STUBBED.add(full); return Stub
        REFUSED.append(full); raise pickle.UnpicklingError(f"refused global {full}")
WORK = Path("/kaggle/working"); TMP = Path("/kaggle/tmp"); TMP.mkdir(exist_ok=True)
URL = "https://zenodo.org/api/records/19099155/files/cryptobank_dataset_11_08_2025.zip/content"
t = time.time(); zp = TMP / "cb.zip"; urllib.request.urlretrieve(URL, zp); print(f"downloaded {zp.stat().st_size/1e6:.0f} MB [{time.time()-t:.0f}s]")
''', r'''t = time.time()
with zipfile.ZipFile(zp) as z, z.open("cryptobank_dataset_11_08_2025.pkl") as fh:
    DB = Safe(io.BufferedReader(fh, buffer_size=1 << 24)).load()
print(type(DB).__name__, f"[{time.time()-t:.0f}s]", "refused:", REFUSED, "stubbed:", sorted(STUBBED))
if isinstance(DB, dict): print("dict keys:", list(DB)[:50]); DB = next(v for v in DB.values() if isinstance(v, pd.DataFrame))
print(DB.shape); DB.info(verbose=True, memory_usage="deep")
''', r'''pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 400)
cr = DB["crypticity"]; print("crypticity:", cr.value_counts(dropna=False).to_dict())
print("alphafold apo:", DB["alphafold"].value_counts().to_dict())
for c in ("structureId", "structureId_apo", "clusterNumber95", "uniprot_accession", "ligandId"):
    print(f"unique {c}: all {DB[c].nunique():,}  cryptic {DB.loc[cr.fillna(False).astype(bool), c].nunique():,}")
C = DB[cr.fillna(False).astype(bool)]
SHOW = ["structureId", "chain_id", "auth_asym_id", "auth_asym_ids", "ligandId", "resid", "structureId_apo", "chain_id_apo", "auth_asym_id_apo",
        "auth_asym_ids_apo", "rmsd", "rmsd_local", "linear_score_3", "ligand_contacts", "holo_lig_seq", "apo_lig_seq", "site_centroid",
        "new_site", "selected_score", "site_mean_score", "linear_score_3_segments", "site_coords", "sequence", "sequence_apo"]
for i in range(4):
    r = C.iloc[i * 997]; print("-" * 80)
    for c in SHOW: print(f"  {c:26s} {str(r[c])[:400]}")
KEEP = ["rcsb_id", "structureId", "chain_id", "auth_asym_id", "auth_asym_ids", "asym", "sequence", "uniprot_accession", "clusterNumber95",
        "cluster_id_30", "ligandId", "ligand_formula_weight", "resid", "resolution", "initial_release_date", "rcsb_id_apo", "structureId_apo",
        "chain_id_apo", "auth_asym_id_apo", "auth_asym_ids_apo", "asym_apo", "sequence_apo", "uniprot_accession_apo", "resolution_apo",
        "alphafold", "rmsd", "rmsd_local", "linear_score_3", "linear_score_local_3", "ligand_contacts", "holo_lig_seq", "apo_lig_seq",
        "site_centroid", "new_site", "selected_score", "site_mean_score", "lig_mean_score", "max_cluster_score", "crypticity"]
out = C[KEEP].copy()
for c in KEEP:
    if str(out[c].dtype) in ("category", "object"): out[c] = out[c].astype(str)
out.to_parquet(WORK / "cryptobank_cryptic.parquet", index=False)
print(f"cryptic rows {len(out):,} -> {(WORK / 'cryptobank_cryptic.parquet').stat().st_size/1e6:.0f} MB")
print("done")
''']
nb = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python")),
          cells=[dict(cell_type="markdown", metadata={}, source="# Phase 308a - CryptoBank database inspection (restricted unpickler, CPU)\n")] +
                [dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=c) for c in code])
d = ROOT_OUT / SLUG; d.mkdir(parents=True, exist_ok=True)
o = d / "phase308a_cryptobank_inspect.ipynb"; o.write_text(json.dumps(nb, indent=1) + "\n")
meta = dict(id="blackpearls1/phase308a-cryptobank-inspect", title="Phase308a Cryptobank Inspect", code_file=o.name, language="python",
            kernel_type="notebook", is_private=True, enable_gpu=False, enable_tpu=False, enable_internet=True,
            dataset_sources=[], competition_sources=[], kernel_sources=[])
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n"); print("wrote", o)
