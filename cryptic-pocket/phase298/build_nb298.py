#!/usr/bin/env python3
"""Phase 298: two more frozen protein language models on top of the confirmed stack (ESM2-3B input + ESM2-650M + ProtT5-XL branches),
on the Phase 295 CV instrument, ALL arms in one kernel (GPU runs are not bitwise reproducible, Phase 296).
  Ankh-large (ElnaggarLab/ankh-large, T5 encoder, 1536-d): input ids built from its single-letter vocabulary + </s>; its tokenizer's
      is_split_into_words path inserts an <unk> before every residue (found locally), so it is not used.
  SaProt-650M-PDB (westlake-repl/SaProt_650M_PDB, ESM architecture, 1280-d): structure-aware tokens = amino acid + Foldseek 3Di letter
      per residue; 3Di from `foldseek structureto3didescriptor` on each apo chain written with the Phase 48 residue rule (verified locally:
      Foldseek returns exactly our residues); any residue it misses gets the '#' (no structure) token.
Arms: stack2 (comparator), stack2_ankh, stack2_saprot, stack4 (both). 8 seeds x 4 folds, fixed 12 epochs.
PRE-REGISTERED: PRIMARY = pooled held-out PR-AUC over 760 chains, stack2_ankh vs stack2 and stack2_saprot vs stack2, paired by seed,
t AND Wilcoxon < .05 after Holm over the two. Secondary: stack4 vs stack2; ROC, per-protein AP, P@20.
usage: build_nb298.py <output_root>"""
import json, subprocess, sys, tempfile
from pathlib import Path

ROOT_OUT = Path(sys.argv[1]).resolve(); SLUG = "298"
HERE = Path(__file__).resolve().parent
tmp = Path(tempfile.mkdtemp())
subprocess.run([sys.executable, str(HERE.parent / "phase296" / "build_nb296.py"), str(tmp)], check=True, capture_output=True)
nb = json.load(open(tmp / "296b" / "phase296b_prott5_cv.ipynb"))
cells = nb["cells"]
src = lambda i: cells[i]["source"] if isinstance(cells[i]["source"], str) else "".join(cells[i]["source"])
def setsrc(i, s): cells[i]["source"] = s
def rep(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)
find = lambda key: next(i for i in range(len(cells)) if key in src(i))

setsrc(0, "# Phase 298 - Ankh-large and SaProt-650M (structure-aware) as further frozen branches, Phase 295 CV instrument, all arms in one kernel\n")
i = find("ProtT5 embeddings for")
setsrc(i, src(i) + r'''

# ---- Ankh-large ----
ankh_tok = AutoTokenizer.from_pretrained("ElnaggarLab/ankh-large")
ankh = T5EncoderModel.from_pretrained("ElnaggarLab/ankh-large").to(DEVICE).eval()
A_IDS = {c: ankh_tok.convert_tokens_to_ids(c) for c in "ACDEFGHIKLMNPQRSTVWY"}; assert ankh_tok.unk_token_id not in A_IDS.values()
@torch.no_grad()
def ankh_embed(seq):
    n = -(-len(seq) // CHUNK); size = -(-len(seq) // n); outs = []
    for i in range(0, len(seq), size):
        part = seq[i:i + size]
        ids = torch.tensor([[A_IDS[c] for c in part] + [ankh_tok.eos_token_id]], device=DEVICE)
        outs.append(ankh(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[0, :len(part)].float())
    return torch.cat(outs)
t_ = time.time()
for d in dataset:
    d.plm3_emb = ankh_embed(seq_of(str(d.pdb_id), str(d.chain_id))).half().cpu(); assert d.plm3_emb.shape == (d.num_nodes, 1536)
del ankh; torch.cuda.empty_cache(); print(f"Ankh embeddings [{(time.time()-t_)/60:.1f}m]", flush=True)

# ---- Foldseek 3Di + SaProt ----
import subprocess
from Bio.PDB import PDBIO, Select
from Bio import Align
FSD = Path("/tmp/fs"); FSD.mkdir(exist_ok=True); FSIN = FSD / "in"; FSIN.mkdir(exist_ok=True)
FS = None
for flavour in ("avx2", "sse2"):
    subprocess.run(f"wget -q https://mmseqs.com/foldseek/foldseek-linux-{flavour}.tar.gz -O {FSD}/fs.tgz && tar xzf {FSD}/fs.tgz -C {FSD}", shell=True)
    if subprocess.run([str(FSD / "foldseek/bin/foldseek"), "version"], capture_output=True).returncode == 0: FS = str(FSD / "foldseek/bin/foldseek"); break
assert FS, "foldseek unavailable"; print("foldseek", flavour)
for d in dataset:
    pid, cid = str(d.pdb_id), str(d.chain_id); path = paths[pid]
    parser = MMCIFParser(QUIET=True) if ".cif" in path.name else PDBParser(QUIET=True, PERMISSIVE=True)
    with gzip.open(path, "rt") as fh: st = parser.get_structure(pid, fh)
    m0 = next(st.get_models()); keep = {id(r) for r in m0[cid] if r.get_resname().strip().upper() in AA1 and "CA" in r}
    class Sel(Select):
        def accept_model(self, mo): return mo is m0
        def accept_chain(self, ch): return ch.id == cid
        def accept_residue(self, r): return id(r) in keep
    try:
        io = PDBIO(); io.set_structure(st); io.save(str(FSIN / f"{pid}_{cid}.pdb"), Sel())
    except Exception as e: print("PDB write failed (chain gets '#' tokens):", pid, cid, type(e).__name__)
subprocess.run([FS, "structureto3didescriptor", "-v", "0", "--threads", "4", "--chain-name-mode", "1", str(FSIN), str(FSD / "out.tsv")], check=True)
F3 = {}
for line in open(FSD / "out.tsv"):
    f = line.rstrip("\n").split("\t"); F3[f[0].split(".pdb")[0]] = (f[1], f[2])
al = Align.PairwiseAligner(); al.mode = "global"; al.match_score = 2; al.mismatch_score = -1; al.open_gap_score = -6; al.extend_gap_score = -1
n_exact = n_hash = n_res = 0
for d in dataset:
    key = f"{d.pdb_id}_{d.chain_id}"; seq = seq_of(str(d.pdb_id), str(d.chain_id)); tdi = ["#"] * len(seq)
    if key in F3:
        aa, t3 = F3[key]
        if aa == seq: tdi = list(t3.lower()); n_exact += 1
        else:
            a = al.align(seq, aa)[0]
            for (s0, s1), (q0, q1) in zip(*a.aligned):
                for i_, j_ in zip(range(s0, s1), range(q0, q1)):
                    if seq[i_] == aa[j_]: tdi[i_] = t3[j_].lower()
    d.sa_seq = "".join(x + y for x, y in zip(seq, tdi)); n_hash += tdi.count("#"); n_res += len(seq)
print(f"3Di: {n_exact}/{len(dataset)} chains exact, residues without structure token {n_hash}/{n_res}")
sap_tok = AutoTokenizer.from_pretrained("westlake-repl/SaProt_650M_PDB")
sap = EsmModel.from_pretrained("westlake-repl/SaProt_650M_PDB", add_pooling_layer=False).to(DEVICE).eval()
@torch.no_grad()
def saprot_embed(sa):
    L = len(sa) // 2; n = -(-L // CHUNK); size = -(-L // n); outs = []
    for i in range(0, L, size):
        part = sa[2 * i: 2 * (i + size)]; ids = sap_tok(part, return_tensors="pt").to(DEVICE)
        assert ids["input_ids"].shape[1] == len(part) // 2 + 2 and sap_tok.unk_token_id not in ids["input_ids"][0].tolist()
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            outs.append(sap(**ids).last_hidden_state[0, 1:-1].float())
    return torch.cat(outs)
t_ = time.time()
for d in dataset:
    d.plm4_emb = saprot_embed(d.sa_seq).half().cpu(); assert d.plm4_emb.shape == (d.num_nodes, 1280)
    del d.sa_seq
del sap; torch.cuda.empty_cache(); print(f"SaProt embeddings [{(time.time()-t_)/60:.1f}m]", flush=True)
''')
i = find("class EGCL"); m = src(i)
m = rep(m, "plm_dim=None, plm2_dim=None):", "plm_dim=None, plm2_dim=None, plm3_dim=None, plm4_dim=None):")
m = rep(m, "            s.plm2_proj=nn.Linear(plm2_dim,h); nn.init.zeros_(s.plm2_proj.weight); nn.init.zeros_(s.plm2_proj.bias)\n",
        "            s.plm2_proj=nn.Linear(plm2_dim,h); nn.init.zeros_(s.plm2_proj.weight); nn.init.zeros_(s.plm2_proj.bias)\n"
        "        s.plm3_dim, s.plm4_dim = plm3_dim, plm4_dim\n"
        "        if plm3_dim:\n"
        "            s.plm3_proj=nn.Linear(plm3_dim,h); nn.init.zeros_(s.plm3_proj.weight); nn.init.zeros_(s.plm3_proj.bias)\n"
        "        if plm4_dim:\n"
        "            s.plm4_proj=nn.Linear(plm4_dim,h); nn.init.zeros_(s.plm4_proj.weight); nn.init.zeros_(s.plm4_proj.bias)\n")
m = rep(m, "        if s.plm2_dim: h=h+s.plm2_proj(d.plm2_emb.float())\n",
        "        if s.plm2_dim: h=h+s.plm2_proj(d.plm2_emb.float())\n"
        "        if s.plm3_dim: h=h+s.plm3_proj(d.plm3_emb.float())\n"
        "        if s.plm4_dim: h=h+s.plm4_proj(d.plm4_emb.float())\n")
setsrc(i, m)
i = find("def fit_fixed"); t = src(i)
t = rep(t, "batch=16, plm2_dim=None):", "batch=16, plm2_dim=None, plm3_dim=None, plm4_dim=None):")
t = rep(t, "plm_dim=plm_dim, plm2_dim=plm2_dim)", "plm_dim=plm_dim, plm2_dim=plm2_dim, plm3_dim=plm3_dim, plm4_dim=plm4_dim)")
t = rep(t, 'if not k.startswith(("plm_proj", "plm2_proj"))', 'if not k.startswith(("plm_proj", "plm2_proj", "plm3_proj", "plm4_proj"))')
setsrc(i, t)
i = find("ARMS = (")
setsrc(i, r'''
KEEP_SIDE = list(range(TOPO_DIM)); m_ = torch.zeros(SIDE_DIM); m_[KEEP_SIDE] = 1.0
for d in dataset: d.x[:, N_PRE:] = d.x[:, N_PRE:] * m_
ARMS = [("stack2", None, None), ("stack2_ankh", 1536, None), ("stack2_saprot", None, 1280), ("stack4", 1536, 1280)]
T0 = time.time(); BUDGET = 10.5 * 3600; ALL = {}
for name, p3, p4 in ARMS:
    for seed in range(8):
        if time.time() - T0 > BUDGET: print(f"budget reached before {name} seed {seed}"); break
        ck = locate(f"pretrain266_full_seed{seed}.pt"); preds = {}
        for f in range(4):
            model = fit_fixed([d for d in CV if d.fold != f], seed, ck, 1280, "bce", 1.0, plm2_dim=1024, plm3_dim=p3, plm4_dim=p4)
            preds.update(predict(model, [d for d in CV if d.fold == f]))
        ALL.setdefault(name, {})[seed] = preds
        torch.save(ALL, WORK / "phase298_cv_preds.pt")
        pp = np.concatenate([v["p"] for v in preds.values()]); yy = np.concatenate([v["y"] for v in preds.values()])
        print(f"  seed {seed} [{name}] held-out PR {average_precision_score(yy, pp):.4f}  [{(time.time()-T0)/60:.1f}m]", flush=True)
print("done")
''')
d = ROOT_OUT / SLUG; d.mkdir(parents=True, exist_ok=True)
o = d / "phase298_ankh_saprot_cv.ipynb"; o.write_text(json.dumps(nb, indent=1) + "\n")
meta = json.load(open(tmp / "296b" / "kernel-metadata.json"))
meta.update(id="blackpearls1/phase298-ankh-saprot-cv", title="Phase298 Ankh Saprot Cv", code_file=o.name)
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote", o)
