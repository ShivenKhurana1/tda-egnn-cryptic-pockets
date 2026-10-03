"""How many OTHER deposited X-ray structures does each CryptoBench protein have? (the 'structural ensemble' channel used by the
cryptic-site benchmark preprint). Counts per UniProt accession via the RCSB search API, excluding every PDB entry CryptoBench itself uses for that
protein (its apo entry and all of its holo entries). 'no ligand' = entry has zero non-polymer entities (waters excluded by RCSB; ions count)."""
import json, sys, time, urllib.request, collections
import numpy as np
D = json.load(open("../cryptobench_dataset.json"))
F = json.load(open("../phase276/cryptobench_folds.json"))
L = json.load(open("../phase276/phase276a_crypticity_labels.json"))
cry_uni = {v["uniprot"] for v in L.values() if v["label"] == "cryptic"}

def search(acc, extra=None):
    nodes = [dict(type="terminal", service="text", parameters=dict(attribute="rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession", operator="exact_match", value=acc)),
             dict(type="terminal", service="text", parameters=dict(attribute="rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name", operator="exact_match", value="UniProt")),
             dict(type="terminal", service="text", parameters=dict(attribute="exptl.method", operator="exact_match", value="X-RAY DIFFRACTION"))]
    if extra: nodes.append(extra)
    q = dict(query=dict(type="group", logical_operator="and", nodes=nodes), return_type="entry", request_options=dict(return_all_hits=True))
    req = urllib.request.Request("https://search.rcsb.org/rcsbsearch/v2/query", data=json.dumps(q).encode(), headers={"Content-Type": "application/json"})
    for t in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
            return {x["identifier"].lower() for x in json.loads(body)["result_set"]} if body else set()
        except Exception as e:
            time.sleep(2 * (t + 1))
    return None


def search_sorted(acc):
    nodes = [dict(type="terminal", service="text", parameters=dict(attribute="rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession", operator="exact_match", value=acc)),
             dict(type="terminal", service="text", parameters=dict(attribute="rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name", operator="exact_match", value="UniProt")),
             dict(type="terminal", service="text", parameters=dict(attribute="exptl.method", operator="exact_match", value="X-RAY DIFFRACTION"))]
    q = dict(query=dict(type="group", logical_operator="and", nodes=nodes), return_type="entry",
             request_options=dict(return_all_hits=True, sort=[dict(sort_by="rcsb_entry_info.resolution_combined", direction="asc")]))
    req = urllib.request.Request("https://search.rcsb.org/rcsbsearch/v2/query", data=json.dumps(q).encode(), headers={"Content-Type": "application/json"})
    for t in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r: body = r.read()
            return [x["identifier"].lower() for x in json.loads(body)["result_set"]] if body else []
        except Exception: time.sleep(2 * (t + 1))
    return None

if __name__ == '__main__':
    noligand = dict(type="terminal", service="text", parameters=dict(attribute="rcsb_entry_info.nonpolymer_entity_count", operator="equals", value=0))
    rows = []
    pdbs = sorted(set().union(*[F[k] for k in ("test",)]))
    for pdb in pdbs:
        ents = D.get(pdb, [])
        mains = [e for e in ents if e.get("is_main_holo_structure")]
        if not mains: continue
        m = mains[0]; acc = m["uniprot_id"]
        own = {pdb.lower()} | {e["holo_pdb_id"].lower() for e in ents if e["uniprot_id"] == acc}
        allx = search(acc); free = search(acc, noligand)
        if allx is None or free is None: rows.append(dict(pdb=pdb, acc=acc, err=True)); continue
        rows.append(dict(pdb=pdb, chain=m["apo_chain"], acc=acc, cryptic=acc in cry_uni, n_xray=len(allx - own), n_noligand=len(free - own), own=len(own)))
        time.sleep(0.15)
    json.dump(rows, open("entry_counts_test.json", "w"), indent=1)
    ok = [r for r in rows if "err" not in r]
    print(f"{len(ok)} test proteins ({len(rows)-len(ok)} failed)")
    def summ(name, rs):
        a = np.array([r["n_xray"] for r in rs]); b = np.array([r["n_noligand"] for r in rs])
        print(f"{name:22s} n={len(rs):3d} | other X-ray entries: median {np.median(a):.0f}, >=1 {np.mean(a>=1)*100:.0f}%, >=3 {np.mean(a>=3)*100:.0f}%, >=6 {np.mean(a>=6)*100:.0f}% || ligand-free entries: median {np.median(b):.0f}, >=1 {np.mean(b>=1)*100:.0f}%, >=3 {np.mean(b>=3)*100:.0f}%")
    summ("all test proteins", ok); summ("strictly cryptic (16)", [r for r in ok if r["cryptic"]])
