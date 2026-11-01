#!/usr/bin/env python3
"""Phase 308d: ingest homology-clean CryptoBank cryptic chains as EXTRA training graphs in the exact Phase 48 node format
(21 one-hot + 6 dihedral + 2560 ESM2-3B + 34 topology), so the stack3 pipeline can train on them unchanged.
Selection (fixed before seeing any model result):
  * Phase 308b records: experimental apo chain, union of apo_lig_seq over its cryptic rows, >= 3 labelled residues present, 30-1500 residues.
  * Homology: MMseqs2 easy-search (sens 7.5, e < 1e-3, no identity floor) of every candidate against ALL CryptoBench chains (sequences
    decoded from the Phase 48 blob's one-hot block, plus the 308b RCSB/database sequences for the chains not in the blob). A 95% cluster
    is dropped entirely if ANY member has ANY hit, so no extra chain is detectably homologous to a CV, val or test chain.
  * One representative per remaining 95% cluster: most holo partners, then best resolution (CryptoBench keeps one apo per protein).
Featurisation is Phase 48's code (Phase 42 topology featuriser, dihedrals, ESM2-3B fp16, radius graph 8 A / 32 nbrs); topology is
standardised with the Phase 48 blob's own mean/std and column mask, not refitted, so extra chains sit on the same scale.
usage: build_nb308d.py <output_root>"""
import json, sys
from pathlib import Path
ROOT_OUT = Path(sys.argv[1]).resolve(); SLUG = "308d"
HERE = Path(__file__).resolve().parent
nb48 = json.load(open(HERE.parents[1] / "archived scripts" / "phase48_cryptobench_ingestion_esm3b.ipynb"))
s48 = lambda i: "".join(nb48["cells"][i]["source"])
def rep(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)
C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})
md("# Phase 308d - homology-clean CryptoBank cryptic chains as extra training graphs (Phase 48 node format)")
code("!pip install -q torch-geometric==2.6.1 gudhi biopython==1.84")
setup = s48(4)
setup = setup[:setup.index("cb_dataset = fetch_json(")] + r'''
def locate(f):
    c = [p for r in (Path("/kaggle/input"), Path("/kaggle/working")) if r.exists() for p in r.rglob(f)]
    if not c: raise FileNotFoundError(f)
    return max(c, key=lambda p: p.stat().st_size)
BLOB = torch.load(locate("phase48_cryptobench_graphs_esm3b.pt"), map_location="cpu", weights_only=False)
AAB = BLOB["aa_alphabet"]
CBSEQ = {f"{d.pdb_id}_{d.chain_id}".upper(): "".join(AAB[i] for i in d.x[:, :21].argmax(1).tolist()) for d in BLOB["dataset"]}
Z = json.load(gzip.open(locate("cryptobank_apo_labels.json.gz"), "rt"))
for k, s in Z["cryptobench_seqs"].items(): CBSEQ.setdefault(k.upper(), s)
print(f"CryptoBench sequences for homology: {len(CBSEQ)} (blob {len(BLOB['dataset'])})")
CH = {k: v for k, v in Z["chains"].items() if len(v["labels"]) >= 3 and v["seq"] and 30 <= len(v["seq"]) <= 1500 and v["chain"]}
print(f"CryptoBank candidates (>= 3 labels, 30-1500 aa): {len(CH):,} chains in {len({v['cluster95'] for v in CH.values()}):,} clusters")
import subprocess
MMD = Path("/tmp/mm"); MMD.mkdir(exist_ok=True)
subprocess.run(f"wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -O {MMD}/m.tgz && tar xzf {MMD}/m.tgz -C {MMD}", shell=True, check=True)
with open(MMD / "q.fa", "w") as f:
    for k, v in CH.items(): f.write(f">{k}\n{v['seq']}\n")
with open(MMD / "t.fa", "w") as f:
    for k, s in CBSEQ.items(): f.write(f">{k}\n{s}\n")
subprocess.run([str(MMD / "mmseqs/bin/mmseqs"), "easy-search", str(MMD / "q.fa"), str(MMD / "t.fa"), str(MMD / "hits.m8"), str(MMD / "tmp"),
                "-s", "7.5", "-e", "1e-3", "--max-seqs", "300", "--threads", str(os.cpu_count()), "-v", "1"], check=True)
hit = {l.split("\t")[0] for l in open(MMD / "hits.m8")}
bad_clusters = {CH[k]["cluster95"] for k in hit if k in CH}
clean = {k: v for k, v in CH.items() if v["cluster95"] not in bad_clusters and k not in CBSEQ}
print(f"chains with any CryptoBench hit: {len(hit & set(CH)):,}; clusters dropped {len(bad_clusters):,}; clean chains {len(clean):,}")
rep_of = {}
for k, v in sorted(clean.items(), key=lambda kv: (-kv[1]["n_holo"], kv[1]["resolution"] if kv[1]["resolution"] == kv[1]["resolution"] else 9.9, kv[0])):
    rep_of.setdefault(v["cluster95"], k)
units = [dict(apo_id=k.split("_")[0].lower(), apo_chain=clean[k]["chain"], pocket_resnums=set(clean[k]["labels"]), pRMSD=None, split="extra",
              key=k, cluster95=clean[k]["cluster95"], uniprot=clean[k]["uniprot"], n_holo=clean[k]["n_holo"]) for k in rep_of.values()]
print(f"representatives (one per clean 95% cluster): {len(units):,}")
'''
code(setup)
code(s48(6))
parse = rep(s48(8), "split=u[\"split\"], pRMSD=u[\"pRMSD\"],", "split=u[\"split\"], pRMSD=u[\"pRMSD\"], key=u[\"key\"], cluster95=u[\"cluster95\"], uniprot=u[\"uniprot\"], n_holo=u[\"n_holo\"],")
parse = rep(parse, 'for s in ("train","test"):', 'for s in ("extra",):')
parse += "\nfr = [r['n_pocket_found'] / r['n_pocket_labeled'] for r in records]\nprint(f'labelled residues found in the parsed chain: median {np.median(fr):.2f}, share >= 0.8 {np.mean(np.array(fr) >= 0.8):.2f}')\n"
code(parse)
code(s48(10))
topo = s48(12)
topo = rep(topo, '''keep = topo_cat.std(axis=0) > 1e-9
dropped = [TOPO_NAMES[i] for i in range(TOPO_DIM) if not keep[i]]
names = [TOPO_NAMES[i] for i in range(TOPO_DIM) if keep[i]]
print(f"pruned {len(dropped)} constant column(s): {dropped}")
mu, sd = topo_cat[:, keep].mean(0), topo_cat[:, keep].std(0); sd[sd < 1e-6] = 1.0''',
'''dropped = list(BLOB["dropped_names"]); keep = np.array([n not in dropped for n in TOPO_NAMES]); names = list(BLOB["topo_names"])
assert [TOPO_NAMES[i] for i in range(TOPO_DIM) if keep[i]] == names and int(keep.sum()) == BLOB["topo_dim"]
mu, sd = np.asarray(BLOB["topo_mean"]), np.asarray(BLOB["topo_std"])   # Phase 48 statistics, not refitted
z_ = (topo_cat[:, keep] - mu) / sd
print(f"extra-chain topology on the Phase 48 scale: mean |z| per column max {np.abs(z_.mean(0)).max():.2f}, std range {z_.std(0).min():.2f}-{z_.std(0).max():.2f}")''')
code(topo)
code(s48(14))
code(s48(16))
g = s48(18)
g = rep(g, 'd.pdb_id, d.chain_id, d.split = r["pdb_id"], r["chain_id"], r["split"]',
        'd.pdb_id, d.chain_id, d.split = r["pdb_id"], r["chain_id"], r["split"]; d.cb_key, d.cluster95, d.uniprot = r["key"], r["cluster95"], r["uniprot"]')
g = rep(g, 'for s in ("train","test"):', 'for s in ("extra",):')
code(g)
code(r'''
blob_x = BLOB["dataset"][0].x.shape[1]
for d in dataset:
    assert d.x.shape[1] == blob_x and torch.isfinite(d.x).all() and torch.isfinite(d.pos).all()
    assert d.edge_index.max() < d.num_nodes and d.edge_index.min() >= 0
print(f"integrity OK; node dim {blob_x} matches the Phase 48 blob")
p = WORK / "phase308d_cryptobank_graphs.pt"
torch.save({"dataset": dataset, "n_pre_topo": N_PRE, "topo_dim": int(keep.sum()), "topo_names": names, "topo_mean": mu, "topo_std": sd,
            "source": "CryptoBank (Martinez et al., Zenodo 19099155, CC-BY-4.0), homology-clean vs CryptoBench"}, p)
json.dump([dict(key=d.cb_key, pdb_id=d.pdb_id, chain=d.chain_id, cluster95=d.cluster95, uniprot=d.uniprot, n=int(d.num_nodes), pos=int(d.y.sum()))
           for d in dataset], open(WORK / "phase308d_manifest.json", "w"), indent=0)
print(f"saved {len(dataset)} graphs -> {p} ({p.stat().st_size/1e6:.0f} MB)")
''')
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
d = ROOT_OUT / SLUG; d.mkdir(parents=True, exist_ok=True)
o = d / "phase308d_cryptobank_graphs.ipynb"; o.write_text(json.dumps(nb, indent=1) + "\n")
meta = dict(id="blackpearls1/phase308d-cryptobank-graphs", title="Phase308d Cryptobank Graphs", code_file=o.name, language="python", kernel_type="notebook",
            is_private=True, enable_gpu=True, machine_shape="NvidiaTeslaT4", enable_tpu=False, enable_internet=True, dataset_sources=[], competition_sources=[],
            kernel_sources=["blackpearls1/tda-egnn-phase48-cryptobench-esm3b", "blackpearls1/phase308b-cryptobank-extract"])
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n"); print("wrote", o)
