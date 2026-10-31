#!/usr/bin/env python3
"""Phase 307: ProstT5 (Rostlab, bilingual amino-acid / Foldseek-3Di T5) as frozen branches on stack3, gated by the Phase 306
deployment-matched CV instrument. SaProt (AA+3Di tokens) was the largest single gain (Phase 298); ProstT5 is a second, differently
trained structure-aware model and can read the 3Di string alone, a pure local-structure view no current branch has.
  prost3di  ProstT5 encoder on "<fold2AA>" + the apo chain's 3Di string (same Foldseek tokens as SaProt; '#' residues -> <unk>)
  prostaa   ProstT5 encoder on "<AA2fold>" + the amino-acid sequence
Token ids are built from the vocab (one token per residue, verified locally: AA = '▁X' pieces, 3Di = lowercase letters).
Each enters through a zero-initialised Linear(1024, h) added to the EGNN's first hidden state, like every other PLM branch.
Arms: stack3 (comparator), stack3_prost3di, stack3_prostaa, stack3_prostboth; 8 seeds x 4 folds, inner early stopping (Phase 306).
Ankh, holo-state and hotspot cells inherited from Phase 303 are removed (unused here; ~25 min of kernel time).
PRE-REGISTERED: PRIMARY = pooled held-out PR-AUC over the 760 CV chains, each arm vs stack3, paired by seed, t AND Wilcoxon < .05
after Holm over the three. Secondary: per-protein AUC/AP overall, on the cryptic-enriched chains (Phase 300) and on the rest.
usage: build_nb307.py <output_root>"""
import json, subprocess, sys, tempfile
from pathlib import Path
ROOT_OUT = Path(sys.argv[1]).resolve(); SLUG = "307"
HERE = Path(__file__).resolve().parent
tmp = Path(tempfile.mkdtemp())
subprocess.run([sys.executable, str(HERE.parent / "phase306" / "build_nb306.py"), str(tmp)], check=True, capture_output=True)
nb = json.load(open(tmp / "306" / "phase306_deploy_cv.ipynb")); cells = nb["cells"]
src = lambda i: cells[i]["source"] if isinstance(cells[i]["source"], str) else "".join(cells[i]["source"])
def rep(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)
find = lambda key: next(i for i in range(len(cells)) if key in src(i))
cells[0]["source"] = "# Phase 307 - ProstT5 (3Di and amino-acid views) as frozen branches on stack3, deployment-matched CV instrument\n"

i = find("# ---- Ankh-large ----"); s = src(i)
a, b = s.index("# ---- Ankh-large ----"), s.index("# ---- Foldseek 3Di + SaProt ----"); s = s[:a] + s[b:]
s = s[:s.index("# ---- holo state")] + r'''# ---- ProstT5: 3Di view and amino-acid view ----
pt_tok = T5Tokenizer.from_pretrained("Rostlab/ProstT5", do_lower_case=False); PV = pt_tok.get_vocab()
pt = T5EncoderModel.from_pretrained("Rostlab/ProstT5", torch_dtype=torch.float16).to(DEVICE).eval()
LET = "ACDEFGHIKLMNPQRSTVWY"
PID_AA = {c: PV["▁" + c] for c in LET}; PID_3D = {c: PV[c] for c in LET.lower()}
assert pt_tok.unk_token_id not in PID_AA.values() and pt_tok.unk_token_id not in PID_3D.values()
@torch.no_grad()
def prost_embed(toks, prefix, table):
    n = -(-len(toks) // CHUNK); size = -(-len(toks) // n); outs = []
    for i in range(0, len(toks), size):
        part = toks[i:i + size]
        ids = torch.tensor([[PV[prefix]] + [table.get(c, pt_tok.unk_token_id) for c in part] + [pt_tok.eos_token_id]], device=DEVICE)
        outs.append(pt(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[0, 1:len(part) + 1].float())
    return torch.cat(outs)
t_ = time.time(); n_unk = 0
for d in dataset:
    d.plm3_emb = prost_embed(d.a3di, "<fold2AA>", PID_3D).half().cpu(); n_unk += sum(c not in PID_3D for c in d.a3di)
    d.plm6_emb = prost_embed(seq_of(str(d.pdb_id), str(d.chain_id)), "<AA2fold>", PID_AA).half().cpu()
    assert d.plm3_emb.shape == d.plm6_emb.shape == (d.num_nodes, 1024) and torch.isfinite(d.plm3_emb).all() and torch.isfinite(d.plm6_emb).all()
    del d.a3di
del pt; torch.cuda.empty_cache(); print(f"ProstT5 3Di + AA embeddings for {len(dataset)} chains, 3Di <unk> residues {n_unk} [{(time.time()-t_)/60:.1f}m]", flush=True)
'''
cells[i]["source"] = s

i = find("class EGCL"); m = src(i)
m = rep(m, "aux3di=False, auxemb=False, hot=False):", "aux3di=False, auxemb=False, hot=False, plm6_dim=None):")
m = rep(m, "        if use_loc_head:\n            s.loc_head", "        s.plm6_dim = plm6_dim\n        if plm6_dim:\n            s.plm6_proj=nn.Linear(plm6_dim,h); nn.init.zeros_(s.plm6_proj.weight); nn.init.zeros_(s.plm6_proj.bias)\n        if use_loc_head:\n            s.loc_head")
m = rep(m, "        if s.hot: h=h+s.hot_proj(d.hot)\n", "        if s.hot: h=h+s.hot_proj(d.hot)\n        if s.plm6_dim: h=h+s.plm6_proj(d.plm6_emb.float())\n")
cells[i]["source"] = m

i = find("ARMS = [("); t = src(i)
t = rep(t, '("plm_proj", "plm2_proj", "plm4_proj", "hemb_head", "hot_proj")', '("plm_proj", "plm2_proj", "plm3_proj", "plm4_proj", "plm6_proj")')
a, b = t.index("ARMS = [("), t.index("T0 = time.time()")
t = t[:a] + '''S3 = dict(plm_dim=1280, plm2_dim=1024, plm4_dim=1280)
ARMS = [("stack3", S3, {}), ("stack3_prost3di", dict(S3, plm3_dim=1024), {}), ("stack3_prostaa", dict(S3, plm6_dim=1024), {}),
        ("stack3_prostboth", dict(S3, plm3_dim=1024, plm6_dim=1024), {})]
''' + t[b:]
t = rep(t, 'WORK / "phase306_cv_preds.pt"', 'WORK / "phase307_cv_preds.pt"')
cells[i]["source"] = t

d = ROOT_OUT / SLUG; d.mkdir(parents=True, exist_ok=True)
o = d / "phase307_prostt5_cv.ipynb"; o.write_text(json.dumps(nb, indent=1) + "\n")
meta = json.load(open(tmp / "306" / "kernel-metadata.json"))
meta.update(id="blackpearls1/phase307-prostt5-cv", title="Phase307 Prostt5 Cv", code_file=o.name)
meta["dataset_sources"] = [x for x in meta.get("dataset_sources", []) if "hotspot" not in x]
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n"); print("wrote", o)
