#!/usr/bin/env python3
"""Phase 311: the final model (stack3, Phase 305/310 protocol, 8 seeds) on Phase 277's two external sets, so the external-validation
figure uses the headline model instead of Phase 266c. PocketMiner's 35-protein MD-derived test set (Phase 277a graphs) and 99 entries of
the independent cryptic-site benchmark (Phase 277b graphs) are re-attached to their structures (same sources and residue rules as
277a/277b; every chain's parsed sequence must equal its graph's one-hot sequence) so the frozen ESM2-650M, ProtT5 and SaProt branches can
embed them. They are never trained on, never used for early stopping, and never used for any choice. Training is identical to
Phase 310 seeds 0-7, so val/test predictions should reproduce it.
usage: build_nb311.py <output_root>"""
import json, subprocess, sys, tempfile
from pathlib import Path
ROOT_OUT = Path(sys.argv[1]).resolve(); SLUG = "311"
HERE = Path(__file__).resolve().parent
tmp = Path(tempfile.mkdtemp())
subprocess.run([sys.executable, str(HERE.parent / "phase310" / "build_nb310.py"), str(tmp)], check=True, capture_output=True)
nb = json.load(open(tmp / "310" / "phase310_stack3_16seed_test.ipynb")); cells = nb["cells"]
src = lambda i: cells[i]["source"] if isinstance(cells[i]["source"], str) else "".join(cells[i]["source"])
def rep(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)
find = lambda key: next(i for i in range(len(cells)) if key in src(i))
cells[0]["source"] = "# Phase 311 - final model (stack3) on the Phase 277 external sets (PocketMiner, cryptic-site benchmark)\n"

i = find("fold_of = {pid: int(k.split"); cells[i]["source"] = src(i) + r'''

# ---- Phase 277 external sets: re-attach structures so the frozen language-model branches can embed them ----
import io, gzip as _gz
AAB = blob["aa_alphabet"]; onehot_seq = lambda d: "".join(AAB[j] for j in d.x[:, :21].argmax(1).tolist())
EXT = {"pocketminer": torch.load(locate("phase277_pocketminer_graphs.pt"), map_location="cpu", weights_only=False)["dataset"],
       "csb_cryptic": torch.load(locate("phase277b_csb_cryptic_graphs.pt"), map_location="cpu", weights_only=False)["dataset"]}
PMBASE = "https://raw.githubusercontent.com/Mickdub/gvp/pocket_pred/data/pm-dataset"
pm_ids = [str(x) for x in np.load(io.BytesIO(urllib.request.urlopen(f"{PMBASE}/test_apo_ids_with_chainids.npy", timeout=60).read()))]
pm_files = [f["name"] for f in fetch_json("https://api.github.com/repos/Mickdub/gvp/contents/data/pm-dataset/apo-structures?ref=pocket_pred")]
def pm_match(idc):                                  # Phase 277a match_file, verbatim logic
    pid, ch = idc[:4], idc[4:]
    cands = [f for f in pm_files if f.lower().startswith(pid.lower())]
    exact = [f for f in cands if f[:len(pid) + len(ch)].lower() == (pid + ch).lower()]
    return exact[0] if exact else (cands[0] if len(cands) == 1 else None)
def pm_chain(model, ch_hint):                       # Phase 277a parse_one chain choice, verbatim logic
    chains = list(model); chain = next((c for c in chains if c.id == ch_hint), None) if ch_hint else None
    return chain or max(chains, key=lambda c: sum(1 for r in c if r.get_resname().strip().upper() in AA1 and "CA" in r))
AA1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K",
       "MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M"}
def chain_seq(path, cid):
    parser = MMCIFParser(QUIET=True) if ".cif" in path.name else PDBParser(QUIET=True, PERMISSIVE=True)
    with _gz.open(path, "rt") as fh: model = next(parser.get_structure("x", fh).get_models())
    return "".join(AA1[r.get_resname().strip().upper()] for r in model[cid] if r.get_resname().strip().upper() in AA1 and "CA" in r)
kept = {}
for name, G in EXT.items():
    ok, bad = [], []
    for j, d in enumerate(G):
        try:
            if name == "pocketminer":
                idc = next(x for x in pm_ids if x[:4] == str(d.pdb_id) and (x[4:] or "A") == str(d.chain_id))
                raw = urllib.request.urlopen(f"{PMBASE}/apo-structures/{pm_match(idc)}", timeout=60).read()
                uid = f"pm_{idc}"; p = RAW / f"{uid}.pdb.gz"; p.write_bytes(_gz.compress(raw))
                with _gz.open(p, "rt") as fh: m0 = next(PDBParser(QUIET=True, PERMISSIVE=True).get_structure(uid, fh).get_models())
                cid = pm_chain(m0, idc[4:]).id; d.ext_key = (idc[:4], str(d.chain_id))
            else:
                pid = str(d.pdb_id); _, p = fetch(pid); uid = f"csb{j:03d}_{pid}"; cid = str(d.chain_id); d.ext_key = (pid, cid)
            paths[uid] = p
            if chain_seq(p, cid) != onehot_seq(d): bad.append((str(d.pdb_id), "sequence mismatch")); continue
            d.pdb_id, d.chain_id = uid, cid
            ok.append(d)
        except Exception as e: bad.append((str(d.pdb_id), type(e).__name__))
    assert len(ok) >= 0.9 * len(G), (name, bad)
    for d in ok:
        assert d.x.size(1) == dataset[0].x.size(1)
        d.fold, d.split, d.y_main = -1, name, d.y.clone()
    kept[name] = ok; splits[name] = ok
    print(f"external {name}: {len(ok)}/{len(G)} chains re-attached with an exact sequence match; failures {bad}")
for name in kept: dataset.extend(kept[name])
'''

i = find("ARMS = [("); t = src(i)
t = rep(t, 'ARMS = [("stack3", {})]', '''ARMS = [("stack3", {})]

@torch.no_grad()
def predict_ext(model, name):
    out = {}
    for d in splits[name]:
        b = d.clone().to(DEVICE); lg, _ = model(b)
        rec = dict(p=torch.sigmoid(lg).cpu().numpy().astype(np.float32), y=d.y.numpy().astype(np.int8))
        if hasattr(d, "eval_mask"): rec["eval_mask"] = d.eval_mask.numpy().astype(bool)
        out[d.ext_key] = rec
    return out''')
t = rep(t, "for seed in range(16):", "for seed in range(8):")
t = rep(t, '"test": predict_split(model, "test"),', '"test": predict_split(model, "test"), "pocketminer": predict_ext(model, "pocketminer"), "csb_cryptic": predict_ext(model, "csb_cryptic"),')
t = rep(t, 'WORK / "phase310_test_preds.pt"', 'WORK / "phase311_ext_preds.pt"')
cells[i]["source"] = t

d = ROOT_OUT / SLUG; d.mkdir(parents=True, exist_ok=True)
o = d / "phase311_stack3_external.ipynb"; o.write_text(json.dumps(nb, indent=1) + "\n")
meta = json.load(open(tmp / "310" / "kernel-metadata.json"))
meta.update(id="blackpearls1/phase311-stack3-external", title="Phase311 Stack3 External", code_file=o.name)
meta["kernel_sources"] = meta["kernel_sources"] + ["blackpearls1/phase277a-pocketminer-featurize", "blackpearls1/phase277b-csb-cryptic-featurize"]
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n"); print("wrote", o)
