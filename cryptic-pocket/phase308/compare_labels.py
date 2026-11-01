"""Phase 308c: how good are CryptoBank's cryptic labels by CryptoBench's standard, and how much NEW data survives homology filtering?
  (1) On apo chains present in both: CryptoBank union label (apo_lig_seq keys over its cryptic rows) vs the CryptoBench official union
      label (apo_pocket_selection over all holo pairings, Phase 291): recall of CryptoBench residues, precision, Jaccard.
  (2) Yield: experimental apo chains, 95% clusters and UniProt IDs left after excluding anything homologous to ANY CryptoBench chain
      (train, val or test), at several identity/coverage cutoffs; positives per chain.
usage: python3 compare_labels.py <cryptobank_apo_labels.json.gz> <cryptobench dataset.json>"""
import gzip, json, sys
import numpy as np
Z = json.load(gzip.open(sys.argv[1], "rt")); CH = Z["chains"]; CB = json.load(open(sys.argv[2]))
off = {}
for pid, ents in CB.items():
    ch = next(e for e in ents if e["is_main_holo_structure"])["apo_chain"]
    off[f"{pid.upper()}_{ch}"] = {s.split("_", 1)[1] for e in ents for s in e["apo_pocket_selection"] if s.split("_", 1)[0] == ch}
both = [k for k in off if k in CH and CH[k]["labels"]]
print(f"CryptoBench apo chains: {len(off)}; with a CryptoBank cryptic label: {len(both)}")
rec, prec, jac = [], [], []
for k in both:
    a, b = off[k], set(CH[k]["labels"])
    if not a: continue
    rec.append(len(a & b) / len(a)); prec.append(len(a & b) / max(len(b), 1)); jac.append(len(a & b) / len(a | b))
q = lambda x: f"mean {np.mean(x):.2f}  median {np.median(x):.2f}  share >= .5: {np.mean(np.array(x) >= .5):.2f}"
print(f"  recall of CryptoBench residues   {q(rec)}\n  precision                        {q(prec)}\n  Jaccard                          {q(jac)}")
print(f"  share with zero overlap: {np.mean(np.array(rec) == 0):.2f}")
lab = {k: v for k, v in CH.items() if v["labels"]}
print(f"\nCryptoBank experimental apo chains with cryptic labels: {len(lab):,} (95% clusters {len({v['cluster95'] for v in lab.values()}):,}, "
      f"UniProt {len({v['uniprot'] for v in lab.values()}):,})")
for ident, cov in ((0.9, 0.5), (0.5, 0.5), (0.3, 0.5), (0.3, 0.3), (0.0, 0.0)):
    keep = {k: v for k, v in lab.items() if (v["cb_hit"] is None if ident == 0 else not (v["cb_ident"] >= ident and max(v["cb_qcov"], v["cb_tcov"]) >= cov))}
    npos = [len(v["labels"]) for v in keep.values()]; L = [len(v["seq"] or "") for v in keep.values()]
    print(f"  exclude hits with ident >= {ident:.1f} & cov >= {cov:.1f}" if ident > 0 else "  exclude ANY e<1e-3 hit         ",
          f"-> chains {len(keep):,}, clusters {len({v['cluster95'] for v in keep.values()}):,}, UniProt {len({v['uniprot'] for v in keep.values()}):,}, "
          f"positives/chain median {np.median(npos) if npos else 0:.0f}, pos rate {np.sum(npos)/max(np.sum(L),1):.3f}")
for name, f in (("max_selected", "max_selected"), ("max_rmsd_local", "max_rmsd_local"), ("n_holo", "n_holo")):
    x = np.array([v[f] for v in lab.values()], float); print(f"  {name:15s} quartiles {np.nanpercentile(x, [25, 50, 75]).round(2).tolist()}")
