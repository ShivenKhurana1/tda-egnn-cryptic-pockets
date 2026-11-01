#!/usr/bin/env python3
"""Phase 308b: extract CryptoBank (Zenodo 19099155) cryptic labels per EXPERIMENTAL apo chain, on a Kaggle CPU kernel, with the same
restricted unpickler as 308a (pandas/numpy globals only; RDKit objects become inert stubs; anything else refused).
Phase 308a showed: 5,989,860 apo-holo-ligand rows; site-level `crypticity` flag (1,800,790 True); `apo_lig_seq` = {auth resnum: resname}
of apo residues around the superposed ligand, i.e. the same kind of label as CryptoBench's apo_pocket_selection.
Per apo chain (structureId_apo, alphafold == False, crypticity True): label = union of apo_lig_seq keys over all its cryptic rows (the
Phase 291 union convention), plus provenance (holo partners, ligands, scores, RMSDs). Homology: mmseqs2 easy-search of every apo
sequence against all 1,107 CryptoBench apo chains (train, val AND test, so neither the CV folds nor the test set can leak); the best hit's
identity and coverage are stored per chain, and filtering happens downstream. CryptoBench sequences come from the database where the apo
chain is present, otherwise from RCSB FASTA.
usage: build_nb308b.py <output_root>"""
import json, subprocess, sys, tempfile
from pathlib import Path
ROOT_OUT = Path(sys.argv[1]).resolve(); SLUG = "308b"
HERE = Path(__file__).resolve().parent
tmp = Path(tempfile.mkdtemp())
subprocess.run([sys.executable, str(HERE / "build_nb308a.py"), str(tmp)], check=True, capture_output=True)
nb = json.load(open(tmp / "308a" / "phase308a_cryptobank_inspect.ipynb")); cells = nb["cells"]
src = lambda c: c["source"] if isinstance(c["source"], str) else "".join(c["source"])
code = [c for c in cells if c["cell_type"] == "code"]
assert "class Safe" in src(code[0]) and "Safe(io.BufferedReader" in src(code[1])
load = src(code[1]).replace("DB.info(verbose=True, memory_usage=\"deep\")", "")
NEW = [load + r'''
COLS = ["structureId_apo", "auth_asym_id_apo", "rcsb_id_apo", "clusterNumber95_apo", "uniprot_accession_apo", "resolution_apo", "sequence_apo",
        "apo_lig_seq", "structureId", "ligandId", "rmsd", "rmsd_local", "selected_score", "site_mean_score", "max_cluster_score", "alphafold", "crypticity"]
ALL_APO = set(map(str, DB["structureId_apo"].cat.categories))
R = DB.loc[(~DB["alphafold"].astype(bool)) & DB["crypticity"].fillna(False).astype(bool), COLS].copy()
for c in R.columns:
    if str(R[c].dtype) == "category": R[c] = R[c].cat.remove_unused_categories()
print(f"experimental-apo cryptic rows {len(R):,}; apo chains {R['structureId_apo'].nunique():,}")
''', r'''cb = json.load(urllib.request.urlopen("https://osf.io/download/ta2ju/"))
CB = {f"{pid.upper()}_{next(e for e in v if e['is_main_holo_structure'])['apo_chain']}": pid for pid, v in cb.items()}
present = [k for k in CB if k in ALL_APO]; print(f"CryptoBench apo chains present in CryptoBank (any row): {len(present)}/{len(CB)}")
sub = DB.loc[DB["structureId_apo"].isin(present), ["structureId_apo", "sequence_apo"]].drop_duplicates("structureId_apo")
CBSEQ = {str(a): str(s) for a, s in zip(sub["structureId_apo"], sub["sequence_apo"])}
def rcsb_seq(key):
    pid, ch = key.split("_")
    try: txt = urllib.request.urlopen(f"https://www.rcsb.org/fasta/entry/{pid}", timeout=30).read().decode()
    except Exception: return None
    for hdr, seq in zip(txt.split("\n")[0::2], txt.split("\n")[1::2]):
        chains = hdr.split("|")[1] if "|" in hdr else ""
        ids = {x.strip().split("[auth ")[-1].rstrip("]") for x in chains.replace("Chains ", "").replace("Chain ", "").split(",")}
        if ch in ids: return seq.strip()
    return None
from concurrent.futures import ThreadPoolExecutor
miss = [k for k in CB if k not in CBSEQ]
with ThreadPoolExecutor(16) as ex:
    for k, s in zip(miss, ex.map(rcsb_seq, miss)):
        if s: CBSEQ[k] = s
print(f"CryptoBench sequences: {len(CBSEQ)}/{len(CB)} (database {len(CB)-len(miss)}, RCSB {len(CBSEQ)-(len(CB)-len(miss))})")
del DB; import gc; gc.collect()
''', r'''t = time.time()
LAB = {}
for cat in R["apo_lig_seq"].cat.categories:
    try: LAB[cat] = set(json.loads(cat))
    except Exception: LAB[cat] = set()
codes = {c: (R[c].cat.codes.to_numpy(), list(map(str, R[c].cat.categories))) for c in ("structureId_apo", "auth_asym_id_apo", "uniprot_accession_apo",
                                                                                    "sequence_apo", "apo_lig_seq", "structureId", "ligandId")}
num = {c: R[c].to_numpy(dtype=float, na_value=np.nan) for c in ("rmsd", "rmsd_local", "selected_score", "site_mean_score", "max_cluster_score",
                                                                 "resolution_apo", "clusterNumber95_apo")}
order = np.argsort(codes["structureId_apo"][0], kind="stable"); ka = codes["structureId_apo"][0][order]
bounds = np.flatnonzero(np.diff(ka)) + 1; groups = np.split(order, bounds)
val = lambda c, i: codes[c][1][codes[c][0][i]] if codes[c][0][i] >= 0 else None
OUT = {}
for g in groups:
    i0 = g[0]; key = val("structureId_apo", i0); labs = set()
    for i in g: labs |= LAB.get(val("apo_lig_seq", i), set())
    OUT[key] = dict(chain=val("auth_asym_id_apo", i0), cluster95=None if np.isnan(num["clusterNumber95_apo"][i0]) else int(num["clusterNumber95_apo"][i0]),
                    uniprot=val("uniprot_accession_apo", i0), resolution=float(num["resolution_apo"][i0]), seq=val("sequence_apo", i0),
                    labels=sorted(labs, key=lambda x: (len(x), x)), n_rows=int(len(g)), holos=sorted({val("structureId", i) for i in g} - {None})[:50],
                    n_holo=len({codes["structureId"][0][i] for i in g}), ligands=sorted({val("ligandId", i) for i in g} - {None})[:30],
                    max_selected=float(np.nanmax(num["selected_score"][g])), max_site_mean=float(np.nanmax(num["site_mean_score"][g])),
                    max_rmsd_local=float(np.nanmax(num["rmsd_local"][g])), med_rmsd=float(np.nanmedian(num["rmsd"][g])))
print(f"apo chains {len(OUT):,}; with labels {sum(bool(v['labels']) for v in OUT.values()):,} [{time.time()-t:.0f}s]")
''', r'''FS = Path("/kaggle/tmp/mm"); FS.mkdir(exist_ok=True)
subprocess_ok = __import__("subprocess").run(f"wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -O {FS}/m.tgz && tar xzf {FS}/m.tgz -C {FS}", shell=True).returncode == 0
MM = str(FS / "mmseqs/bin/mmseqs"); print("mmseqs ok:", subprocess_ok)
with open(FS / "q.fa", "w") as f:
    for k, v in OUT.items():
        if v["seq"]: f.write(f">{k}\n{v['seq']}\n")
with open(FS / "t.fa", "w") as f:
    for k, s in CBSEQ.items(): f.write(f">{k}\n{s}\n")
import subprocess
subprocess.run([MM, "easy-search", str(FS / "q.fa"), str(FS / "t.fa"), str(FS / "hits.m8"), str(FS / "tmpdir"), "--min-seq-id", "0.0", "-e", "1e-3",
                "--format-output", "query,target,fident,qcov,tcov,evalue", "-s", "7.5", "--max-seqs", "50", "--threads", "4", "-v", "1"], check=True)
best = {}
for line in open(FS / "hits.m8"):
    q, tg, fid, qc, tc, ev = line.split("\t"); fid, qc, tc = float(fid), float(qc), float(tc)
    if q not in best or fid * min(qc, tc) > best[q][1] * min(best[q][2], best[q][3]): best[q] = (tg, fid, qc, tc, float(ev))
for k, v in OUT.items():
    b = best.get(k); v["cb_hit"] = b[0] if b else None; v["cb_ident"] = b[1] if b else 0.0; v["cb_qcov"] = b[2] if b else 0.0; v["cb_tcov"] = b[3] if b else 0.0
    v["cb_split"] = None
print(f"chains with any CryptoBench hit (e<1e-3): {sum(k in best for k in OUT):,}/{len(OUT):,}")
import gzip
json.dump(dict(chains=OUT, cryptobench_seqs=CBSEQ, cryptobench_keys=CB), gzip.open(WORK / "cryptobank_apo_labels.json.gz", "wt"))
print(f"wrote {(WORK / 'cryptobank_apo_labels.json.gz').stat().st_size/1e6:.1f} MB")
print("done")
''']
cells[:] = [dict(cell_type="markdown", metadata={}, source="# Phase 308b - CryptoBank cryptic labels per experimental apo chain + homology to CryptoBench\n"),
            code[0]] + [dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s) for s in NEW]
d = ROOT_OUT / SLUG; d.mkdir(parents=True, exist_ok=True)
o = d / "phase308b_cryptobank_extract.ipynb"; o.write_text(json.dumps(nb, indent=1) + "\n")
meta = json.load(open(tmp / "308a" / "kernel-metadata.json")); meta.update(id="blackpearls1/phase308b-cryptobank-extract", title="Phase308b Cryptobank Extract", code_file=o.name)
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n"); print("wrote", o)
