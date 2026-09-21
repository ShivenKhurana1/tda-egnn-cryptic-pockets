#!/usr/bin/env python3
"""Phase 93: does the ESM2-3B capacity lever move COACH420 past round-3's
plateau (0.5932 DCC, 8-seed, at 35M)? Every cell spliced verbatim from
Phase 20 v2 (same methodology as Phase31/38/53) but loading Phase96's
ESM2-3B blob instead of Phase52's 35M blob. GPU-enabled per standing
instruction (dataset unchanged, only node-feature width grew 541->2621)."""
import json, pickle
from pathlib import Path

C20 = pickle.load(open("/tmp/p20_cells.pkl", "rb"))
s20 = lambda i: (C20[i]["source"] if isinstance(C20[i]["source"], str) else "".join(C20[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 93 — Does ESM2-3B Move COACH420 Past the Round-3 Plateau?

Round-3 data scale-up plateaued: 0.5956 (8-seed, Phase40, 2.6x) ->
0.5932 (8-seed, Phase53-line, 3.65x) -- more DATA stopped helping. This
swaps the embedding model 35M -> 3B (the lever proven on CryptoBench:
monotonic AUC/MCC/AUPRC gains at every step) while holding the training
pool and all other pipeline stages identical to Phase53. 3-seed first
pass, same discipline as every other new arm.
''')

md(r'''## Cell 1 — Setup (loads Phase 96's ESM2-3B blob + Phase 51's topology)''')
code(s20(2).replace('transformers==4.46.3', 'transformers==4.46.3 accelerate'))
code(s20(3).replace('locate("phase10_graphs.pt")', 'locate("phase96_graphs_esm3b.pt")')
           .replace('locate("phase6b_graphs.pt")', 'locate("phase51_graphs.pt")')
           .replace('dataset = blob10["dataset"]',
                     'dataset = blob10["dataset"]\n'
                     'for _d in dataset: _d.x = _d.x.float()   # Phase96 stores fp16 on disk (quota fix); upcast once here'))

md(r'''## Cell 2 — Fetch ALL 420 COACH420 structures (identical to Phase 20/31/38/53)''')
code(s20(5))

md(r'''## Cell 3 — Clean the TRAINING side: exclude any chain homologous to COACH420 (identical to Phase 20 v2/31/38/53)''')
code(s20(7))

md(r'''## Cell 4 — Recover true ligand centers for the cleaned training set (identical to Phase 20/31/38/53/14)''')
code(s20(9))

md(r'''## Cell 5 — Model (identical to every prior localization phase; N_PRE/TOPO_DIM read dynamically so this works unchanged at the new 2621-dim node width)''')
code(s20(11))

md(r'''## Cell 6 — Equivariance check before training anything (identical to Phase 20/31/38/53)''')
code(s20(13))

md(r'''## Cell 7 — Training machinery (identical to Phase 14/20/31/38/53)''')
code(s20(15))

md(r'''## Cell 8 — Train the clean-training ensemble (3 seeds, lambda=0.1) on the ESM2-3B blob''')
code(s20(17).replace("clean_deploy_seed{seed}.pt", "clean_deploy_esm3b_seed{seed}.pt"))

md(r'''## Cell 8b — Free memory AND disk before loading a second ESM2-3B model (v3/v4 both still died at the identical ~30s-into-load point despite freeing dataset/splits/blob10 -- diagnostics added this round since 3 memory-only fixes in a row made zero difference to the failure point, suggesting the real ceiling may be DISK, not RAM: `RAW`/`/tmp/pdb_raw` accumulates BOTH the ~7,086 training-side PDB downloads (Cell 4, no longer needed after true-center recovery) AND the 420 COACH420 files (still needed for Cell 10's dihedral recomputation) -- prune everything except what's still needed)''')
code(r'''
import gc, shutil

def _disk(label):
    for p in ("/", "/tmp", "/kaggle/working"):
        try:
            du = shutil.disk_usage(p)
            print(f"  [{label}] {p}: {du.used/1e9:.1f}GB used / {du.total/1e9:.1f}GB total ({du.free/1e9:.1f}GB free)")
        except Exception as e:
            print(f"  [{label}] {p}: error {e}")

def _mem(label):
    try:
        info = {}
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            info[k] = int(v.strip().split()[0])  # kB
        total = info.get("MemTotal", 0)/1e6; avail = info.get("MemAvailable", 0)/1e6
        print(f"  [{label}] host RAM: {total-avail:.1f}GB used / {total:.1f}GB total ({avail:.1f}GB available)")
    except Exception as e:
        print(f"  [{label}] host RAM: error {e}")
    if torch.cuda.is_available():
        print(f"  [{label}] GPU mem: {torch.cuda.memory_allocated()/1e9:.1f}GB allocated, "
              f"{torch.cuda.memory_reserved()/1e9:.1f}GB reserved")

print("disk/memory usage BEFORE cleanup:")
_disk("before"); _mem("before")

N_TRAIN_CHAINS = len(dataset)
del dataset, splits, blob10
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
print(f"freed training-pool memory ({N_TRAIN_CHAINS:,} chains)")

keep = {str(p.resolve()) for p in coach_paths.values()}
freed_bytes = 0; removed = 0
for f in RAW.iterdir():
    if str(f.resolve()) not in keep:
        try:
            freed_bytes += f.stat().st_size; f.unlink(); removed += 1
        except Exception: pass
print(f"pruned RAW cache: removed {removed:,} files ({freed_bytes/1e9:.2f} GB), "
      f"kept {len(keep)} needed for COACH420 dihedral recomputation")

print("disk/memory usage AFTER cleanup:")
_disk("after"); _mem("after")
''')

md(r'''## Cell 9 — Ground-truth ligands for ALL 420 COACH420 structures (identical to Phase 20/31/38/53)''')
code(s20(19))

md(r'''## Cell 10 — Featurize (dihedral + ESM2-3B + topology; cache filename updated)''')
code(s20(21).replace("cache/phase20_raw_features.pkl", "cache/phase93_raw_features.pkl")
            .replace('AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")',
                      'AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")')
            .replace('AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(DEVICE).eval()',
                      'AutoModel.from_pretrained("facebook/esm2_t36_3B_UR50D", torch_dtype=torch.float16, low_cpu_mem_usage=True).to(DEVICE).eval()  '
                      '# v6: avoid materializing a full fp32 copy before .half() -- v2-v5 all died ~30s into this exact load despite ruling out disk (v5 diagnostics showed 1.1TB free) and despite freeing the training dataset first')
            .replace('out = esm(**enc).last_hidden_state\n    return out[0,1:1+len(seq),:].float().cpu()',
                      'out = esm(**enc).last_hidden_state\n    return out[0,1:1+len(seq),:].float().cpu()  # 3B model is fp16; upcast here')
            .replace('tok = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")',
                      '_disk("pre-ESM-load"); _mem("pre-ESM-load")\n'
                      'print("loading ESM2-3B now...")\n'
                      'tok = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")'))

md(r'''## Cell 11 — P2Rank on the FULL 420 (same-run controlled reference, identical to Phase 20/31/38/53)''')
code(s20(23).replace('Path("/tmp/p2_20")', 'Path("/tmp/p2_93")')
            .replace('Path("/tmp/p2_20_out")', 'Path("/tmp/p2_93_out")'))

md(r'''## Cell 12 — Score both, compare to literature AND to the 35M-vs-3B comparison''')
code(s20(25).replace("phase20_summary.json", "phase93_summary.json")
            .replace('"n_train_clean": len(dataset)},', '"n_train_clean": N_TRAIN_CHAINS},'))

code(r'''
print(f"\n--- COACH420 DCC: ESM2 capacity lever at round-3 scale ---")
print(f"{'embedding':<28}{'hidden':>8}{'DCC (3-seed)':>14}{'DCC (8-seed)':>14}")
print("-"*64)
print(f"{'ESM2-35M (Phase53/56)':<28}{480:>8}{0.5763:>14.4f}{0.5932:>14.4f}")
print(f"{'ESM2-3B (this notebook)':<28}{2560:>8}{ours_dcc_rate:>14.4f}{'pending':>14}")
print(f"\ntraining pool size: {N_TRAIN_CHAINS:,} clean chains (round-3, unchanged)")
print(f"this-run P2Rank DCC: {p2_dcc_rate:.4f}  (deterministic tool, should match prior rounds' ~0.5209)")
''')

md(r'''
---

## Reading this

- **DCC beats 0.5932** -- embedding capacity is a real, addressable lever
  here too (mirrors CryptoBench), worth extending to 8 seeds and then
  applying to HOLO4K (Phase94).
- **DCC roughly matches 0.5932 (within noise)** -- capacity doesn't help
  once topology+dihedral+scale have already saturated what COACH420's
  training signal can support; would narrow the "improve until no further
  improvement" search to whatever's left (noncryptic auxiliary data, or
  conclude the writeup is the remaining next step).
- **DCC drops** -- would be surprising (bigger, better-pretrained embeddings
  making things worse) and worth real scrutiny (e.g. an fp16-precision
  regression) before accepting.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase93_esm3b_coach420.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
