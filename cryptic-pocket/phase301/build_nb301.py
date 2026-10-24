#!/usr/bin/env python3
"""Phase 301: holo-state privileged information for cryptic sites, on the Phase 295 CV instrument (all arms in one kernel).
The ligand-bound (holo) structure exists for every CryptoBench training protein but never at prediction time. Its local structure, as
Foldseek 3Di letters on the main holo chain (mapped to apo residues by sequence alignment), is used three ways:
  oracle_holo  SaProt reads AA + HOLO 3Di instead of apo 3Di, train and held-out alike. Not deployable: the HEADROOM if the open state
               were known.
  stud3di      stack3 + an auxiliary head predicting each residue's holo 3Di letter (20-way cross-entropy, weight 0.3, residues without a
               mapped holo letter masked). Training-time only -> deployable.
  studemb      stack3 + an auxiliary head regressing the holo-state SaProt embedding (1 - cosine, weight 0.3). Training-time only.
Comparator: stack3 = ESM2-3B input + ESM2-650M + ProtT5 + SaProt(apo) branches.
PRE-REGISTERED: PRIMARY = per-protein AUC on the cryptic-enriched CV chains (Phase 300 consensus apo count <= 414; 379 chains), seed-level
means, stud3di and studemb each vs stack3, paired t AND Wilcoxon < .05 after Holm over the two. Secondary: per-protein AP and P@20 on the
cryptic-enriched chains; the same on the rest; pooled held-out PR on all 760. oracle_holo is reported as headroom, not as a candidate.
usage: build_nb301.py <output_root>"""
import json, subprocess, sys, tempfile
from pathlib import Path

ROOT_OUT = Path(sys.argv[1]).resolve(); SLUG = "301"
HERE = Path(__file__).resolve().parent
tmp = Path(tempfile.mkdtemp())
subprocess.run([sys.executable, str(HERE.parent / "phase298" / "build_nb298.py"), str(tmp)], check=True, capture_output=True)
nb = json.load(open(tmp / "298" / "phase298_ankh_saprot_cv.ipynb"))
cells = nb["cells"]
src = lambda i: cells[i]["source"] if isinstance(cells[i]["source"], str) else "".join(cells[i]["source"])
def setsrc(i, s): cells[i]["source"] = s
def rep(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)
find = lambda key: next(i for i in range(len(cells)) if key in src(i))

setsrc(0, "# Phase 301 - holo-state privileged information (oracle holo-SaProt, holo-3Di and holo-embedding auxiliary heads), Phase 295 CV\n")
i = find("SaProt embeddings [")
setsrc(i, rep(src(i), "    del d.sa_seq\n", "    d.a3di = d.sa_seq[1::2]; del d.sa_seq\n") + r'''

# ---- holo state: main holo chain -> Foldseek 3Di -> mapped onto apo residues ----
HIN = FSD / "holo_in"; HIN.mkdir(exist_ok=True)
main_of = {pid: next(e for e in ents if e.get("is_main_holo_structure")) for pid, ents in cb_dataset.items()}
hneed = sorted({main_of[str(d.pdb_id)]["holo_pdb_id"].lower() for d in dataset})
with ThreadPoolExecutor(max_workers=16) as ex: paths.update({p: f for p, f in ex.map(fetch, hneed) if f})
hkey = {}
for d in dataset:
    m = main_of[str(d.pdb_id)]; hp, hc = m["holo_pdb_id"].lower(), m["holo_chain"]
    if hp not in paths: continue
    try:
        path = paths[hp]; parser = MMCIFParser(QUIET=True) if ".cif" in path.name else PDBParser(QUIET=True, PERMISSIVE=True)
        with gzip.open(path, "rt") as fh: st = parser.get_structure(hp, fh)
        m0 = next(st.get_models()); chain = m0[hc]; keep = {id(r) for r in chain if r.get_resname().strip().upper() in AA1 and "CA" in r}
        class SelH(Select):
            def accept_model(self, mo): return mo is m0
            def accept_chain(self, ch): return ch is chain
            def accept_residue(self, r): return id(r) in keep
        if len(hc) > 1: chain.id = next(x for x in "zyxwvutsrqponmlkjihgfedcba" if x not in {c.id for c in m0})
        io = PDBIO(); io.set_structure(st); name = f"{d.pdb_id}_{d.chain_id}__holo"; io.save(str(HIN / f"{name}.pdb"), SelH()); hkey[name] = d
    except Exception as e: print("holo write failed:", d.pdb_id, hp, hc, type(e).__name__)
subprocess.run([FS, "structureto3didescriptor", "-v", "0", "--threads", "4", "--chain-name-mode", "1", str(HIN), str(FSD / "holo.tsv")], check=True)
H3 = {}
for line in open(FSD / "holo.tsv"):
    f = line.rstrip("\n").split("\t"); H3[f[0].split(".pdb")[0]] = (f[1], f[2])
LET = "ACDEFGHIKLMNPQRSTVWY"; L2I = {c: i for i, c in enumerate(LET.lower())}
n_map = n_res = n_diff = n_diff_pos = n_pos_map = 0
for d in dataset:
    seq = seq_of(str(d.pdb_id), str(d.chain_id)); h3 = ["#"] * len(seq); hk = f"{d.pdb_id}_{d.chain_id}__holo"
    if hk in H3:
        aa, t3 = H3[hk]; a = al.align(seq, aa)[0]
        for (s0, s1), (q0, q1) in zip(*a.aligned):
            for i_, j_ in zip(range(s0, s1), range(q0, q1)):
                if seq[i_] == aa[j_]: h3[i_] = t3[j_].lower()
    d.h3di = torch.tensor([L2I.get(c, -1) for c in h3], dtype=torch.long)
    d.hsa_seq = "".join(x + y for x, y in zip(seq, h3))
    mk = d.h3di >= 0; n_map += int(mk.sum()); n_res += len(seq)
    diff = torch.tensor([a_ != b_ for a_, b_ in zip(d.a3di, h3)]) & mk; n_diff += int(diff.sum())
    n_diff_pos += int((diff & (d.y > 0)).sum()); n_pos_map += int((mk & (d.y > 0)).sum())
print(f"holo 3Di mapped for {n_map}/{n_res} residues; apo->holo 3Di letter changes: {n_diff/max(n_map,1):.1%} overall, "
      f"{n_diff_pos/max(n_pos_map,1):.1%} at pocket residues")
sap = EsmModel.from_pretrained("westlake-repl/SaProt_650M_PDB", add_pooling_layer=False).to(DEVICE).eval()
t_ = time.time()
for d in dataset:
    d.plm5_emb = saprot_embed(d.hsa_seq).half().cpu(); assert d.plm5_emb.shape == (d.num_nodes, 1280)
    d.hmask = d.h3di >= 0; del d.hsa_seq, d.a3di
del sap; torch.cuda.empty_cache(); print(f"holo-state SaProt embeddings [{(time.time()-t_)/60:.1f}m]", flush=True)
''')
i = find("class EGCL"); m = src(i)
m = rep(m, "plm3_dim=None, plm4_dim=None):", "plm3_dim=None, plm4_dim=None, oracle=False, aux3di=False, auxemb=False):")
m = rep(m, "        if plm4_dim:\n", "        s.oracle, s.aux3di, s.auxemb, s.aux_out = oracle, aux3di, auxemb, {}\n"
        "        if aux3di: s.h3_head=nn.Sequential(nn.Linear(h,h),nn.SiLU(),nn.Linear(h,20))\n"
        "        if auxemb: s.hemb_head=nn.Sequential(nn.Linear(h,h),nn.SiLU(),nn.Linear(h,1280))\n"
        "        if plm4_dim:\n")
m = rep(m, "        if s.plm4_dim: h=h+s.plm4_proj(d.plm4_emb.float())\n",
        "        if s.plm4_dim: h=h+s.plm4_proj((d.plm5_emb if s.oracle else d.plm4_emb).float())\n")
m = rep(m, "        logits = s.head(h).squeeze(-1)\n", "        logits = s.head(h).squeeze(-1)\n"
        "        s.aux_out = {}\n"
        "        if s.aux3di: s.aux_out['h3'] = s.h3_head(h)\n"
        "        if s.auxemb: s.aux_out['hemb'] = s.hemb_head(h)\n")
setsrc(i, m)
i = find("def run_epoch"); t = src(i)
t = rep(t, "        if lam > 0 and pred_c is not None: loss = loss + lam * huber(pred_c, b.true_center.view(-1, 3))\n",
        "        if lam > 0 and pred_c is not None: loss = loss + lam * huber(pred_c, b.true_center.view(-1, 3))\n"
        "        if 'h3' in model.aux_out and b.hmask.any():\n"
        "            loss = loss + AUX_W * nn.functional.cross_entropy(model.aux_out['h3'][b.hmask], b.h3di[b.hmask])\n"
        "        if 'hemb' in model.aux_out and b.hmask.any():\n"
        "            loss = loss + AUX_W * (1 - nn.functional.cosine_similarity(model.aux_out['hemb'][b.hmask], b.plm5_emb[b.hmask].float(), dim=-1)).mean()\n")
t = rep(t, "plm3_dim=None, plm4_dim=None):\n", "plm3_dim=None, plm4_dim=None, flags=None):\n")
t = rep(t, "plm3_dim=plm3_dim, plm4_dim=plm4_dim).to(DEVICE)", "plm3_dim=plm3_dim, plm4_dim=plm4_dim, **(flags or {})).to(DEVICE)")
t = rep(t, 'if not k.startswith(("plm_proj", "plm2_proj", "plm3_proj", "plm4_proj"))', 'if not k.startswith(("plm_proj", "plm2_proj", "plm3_proj", "plm4_proj", "h3_head", "hemb_head"))')
setsrc(i, t)
i = find('ARMS = [("stack2"')
setsrc(i, r'''
KEEP_SIDE = list(range(TOPO_DIM)); m_ = torch.zeros(SIDE_DIM); m_[KEEP_SIDE] = 1.0
for d in dataset: d.x[:, N_PRE:] = d.x[:, N_PRE:] * m_
AUX_W = 0.3
ARMS = [("stack3", {}), ("oracle_holo", dict(oracle=True)), ("stud3di", dict(aux3di=True)), ("studemb", dict(auxemb=True))]
T0 = time.time(); BUDGET = 10.5 * 3600; ALL = {}
for name, flags in ARMS:
    for seed in range(8):
        if time.time() - T0 > BUDGET: print(f"budget reached before {name} seed {seed}"); break
        ck = locate(f"pretrain266_full_seed{seed}.pt"); preds = {}
        for f in range(4):
            model = fit_fixed([d for d in CV if d.fold != f], seed, ck, 1280, "bce", 1.0, plm2_dim=1024, plm4_dim=1280, flags=flags)
            preds.update(predict(model, [d for d in CV if d.fold == f]))
        ALL.setdefault(name, {})[seed] = preds
        torch.save(ALL, WORK / "phase301_cv_preds.pt")
        pp = np.concatenate([v["p"] for v in preds.values()]); yy = np.concatenate([v["y"] for v in preds.values()])
        print(f"  seed {seed} [{name}] held-out PR {average_precision_score(yy, pp):.4f}  [{(time.time()-T0)/60:.1f}m]", flush=True)
print("done")
''')
d = ROOT_OUT / SLUG; d.mkdir(parents=True, exist_ok=True)
o = d / "phase301_holo_privileged_cv.ipynb"; o.write_text(json.dumps(nb, indent=1) + "\n")
meta = json.load(open(tmp / "298" / "kernel-metadata.json"))
meta.update(id="blackpearls1/phase301-holo-privileged-cv", title="Phase301 Holo Privileged Cv", code_file=o.name)
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote", o)
