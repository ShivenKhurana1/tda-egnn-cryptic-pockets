"""Phase 276a: map the "A benchmark selected for crypticity" (pmfran4252-afk/cryptic-site-benchmark) relabeling of the CryptoBench test split onto our own
(pdb_id, chain_id) test keys, via CryptoBench's own dataset.json (uniprot_id per is_main_holo_structure entry) and folds.json (test-fold pdb ids).
Their rule: cryptic = apo_vox<=30 AND holo_vox>=30 (pocket absent in apo, present in holo); pre-formed = apo_vox>=60 AND holo_vox>=30; else unlabelled (None).
Source: https://github.com/pmfran4252-afk/cryptic-site-benchmark (CC BY 4.0); CryptoBench data from https://osf.io/pz4a9/ (Skrhak et al. 2025, Bioinformatics)."""
import json
d = json.load(open("cryptobench_dataset.json"))
folds = json.load(open("cryptobench_folds.json"))
mapping = {}
for pdb in folds["test"]:
    for e in d.get(pdb, []):
        if e.get("is_main_holo_structure"):
            mapping[(pdb, e["apo_chain"])] = e["uniprot_id"]
crypt = {r["gid"]: r for r in json.load(open("cb_crypticity.json"))}
labels = {}
for (pdb, ch), gid in mapping.items():
    r = crypt.get(gid)
    if r is None: continue
    labels[f"{pdb}_{ch}"] = dict(label=r["label"], apo_vox=r["apo_vox"], holo_vox=r["holo_vox"], uniprot=gid)
json.dump(labels, open("phase276a_crypticity_labels.json", "w"), indent=1)
from collections import Counter
print("mapped", len(mapping), "test (pdb,chain) pairs; labelled", len(labels), Counter(v["label"] for v in labels.values()))
