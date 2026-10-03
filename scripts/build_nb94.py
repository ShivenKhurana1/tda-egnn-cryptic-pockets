#!/usr/bin/env python3
"""Phase 94: does the ESM2-3B capacity lever move HOLO4K past round-3's
result (0.6453 DCC, 8-seed, at 35M, +0.0124-ish parity/slight win vs
P2Rank)? Every cell spliced verbatim from Phase 24 (same methodology as
Phase33/39/54) but loading Phase96's ESM2-3B blob instead of Phase52's
35M blob. GPU-enabled per standing instruction."""
import json, pickle
from pathlib import Path

C24 = pickle.load(open("/tmp/p24_cells.pkl", "rb"))
s24 = lambda i: (C24[i]["source"] if isinstance(C24[i]["source"], str) else "".join(C24[i]["source"]))

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 94 - Does ESM2-3B Move HOLO4K Past the Round-3 Result?

Round-3 data scale-up gave HOLO4K a modest, roughly-holding win/parity
against live P2Rank (0.6453 DCC 8-seed vs P2Rank ~0.6580 on this pool --
essentially a wash). This swaps the embedding model 35M -> 3B (the lever
proven on CryptoBench, and now being tested on COACH420 in Phase93) while
holding the training pool and all other pipeline stages identical to
Phase54. 3-seed first pass, same discipline as every other new arm.
''')

md(r'''## Cell 1 - Setup (loads Phase 96's ESM2-3B blob + Phase 51's topology)''')
code(s24(2).replace('transformers==4.46.3', 'transformers==4.46.3 accelerate'))
code(s24(3).replace('locate("phase10_graphs.pt")', 'locate("phase96_graphs_esm3b.pt")')
           .replace('locate("phase6b_graphs.pt")', 'locate("phase51_graphs.pt")')
           .replace('dataset = blob10["dataset"]',
                     'dataset = blob10["dataset"]\n'
                     'for _d in dataset: _d.x = _d.x.float()   # Phase96 stores fp16 on disk (quota fix); upcast once here'))

md(r'''## Cell 2 - Sample the SAME ~1,000 HOLO4K structures (seed=2024, identical to Phase 24/33/39/54)''')
code(s24(5))

md(r'''## Cell 3 - Clean the TRAINING side (identical logic; pool size 7513, unchanged from Phase54)''')
code(s24(7).replace("2056", "7513"))

md(r'''## Cell 4 - Recover true ligand centers (identical to Phase 24/33/39/54/14)''')
code(s24(9))

md(r'''## Cell 5 - Model (identical to every prior localization phase; N_PRE/TOPO_DIM read dynamically so this works unchanged at the new 2621-dim node width)''')
code(s24(11))

md(r'''## Cell 6 - Training machinery (identical to Phase 24/33/39/54/14)''')
code(s24(13))

md(r'''## Cell 7 - Train the clean ensemble (3 seeds) on the ESM2-3B blob''')
code(s24(15).replace("holo_clean_deploy_seed{seed}.pt", "holo_clean_deploy_esm3b_seed{seed}.pt"))

md(r'''## Cell 7b - Free memory AND disk before loading a second ESM2-3B model (v3/v4 both still died at the identical ~30s-into-load point despite freeing dataset/splits/blob10 -- diagnostics added this round since 3 memory-only fixes in a row made zero difference to the failure point, suggesting the real ceiling may be DISK, not RAM: `RAW`/`/tmp/pdb_raw` accumulates BOTH the ~7,086 training-side PDB downloads (Cell 4, no longer needed after true-center recovery) AND the sampled HOLO4K files (still needed for the featurize cell's dihedral recomputation) -- prune everything except what's still needed)''')
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

keep = {str(p.resolve()) for p in holo_paths.values()}
freed_bytes = 0; removed = 0
for f in RAW.iterdir():
    if str(f.resolve()) not in keep:
        try:
            freed_bytes += f.stat().st_size; f.unlink(); removed += 1
        except Exception: pass
print(f"pruned RAW cache: removed {removed:,} files ({freed_bytes/1e9:.2f} GB), "
      f"kept {len(keep)} needed for HOLO4K dihedral recomputation")

print("disk/memory usage AFTER cleanup:")
_disk("after"); _mem("after")
''')

md(r'''## Cell 8 - Ground truth + featurize the sampled 1,000 structures (identical to Phase 24/33/39/54)''')
code(s24(17))

md(r'''## Cell 9 - Featurize (dihedral + ESM2-3B + topology; cache filename updated)''')
code(s24(19).replace("phase24_raw_features.pkl", "phase94_raw_features.pkl")
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

md(r'''## Cell 10 - P2Rank on the same 1,000 structures (identical to Phase 24/33/39/54)''')
code(s24(21))

md(r'''## Cell 11 - Score, compare to literature AND to the 35M-vs-3B comparison''')
code(s24(23).replace("2056", "7513").replace("phase24_summary.json", "phase94_summary.json")
            .replace("7513-len(dataset)", "7513-N_TRAIN_CHAINS"))

code(r'''
print(f"\n--- HOLO4K DCC: ESM2 capacity lever at round-3 scale ---")
print(f"{'embedding':<28}{'hidden':>8}{'DCC (3-seed)':>14}{'DCC (8-seed)':>14}{'gap to P2Rank':>16}")
print("-"*82)
print(f"{'ESM2-35M (Phase54/56)':<28}{480:>8}{0.6483:>14.4f}{0.6453:>14.4f}{-0.0127:>16.4f}")
print(f"{'ESM2-3B (this notebook)':<28}{2560:>8}{ours_dcc_rate:>14.4f}{'pending':>14}"
      f"{ours_dcc_rate-p2_dcc_rate:>16.4f}")
print(f"\ntraining pool size: {N_TRAIN_CHAINS:,} clean chains (round-3, unchanged)")
print(f"this-run P2Rank DCC: {p2_dcc_rate:.4f}  (deterministic tool, same 1000-structure sample, "
      f"should match prior rounds' ~0.6580)")
''')

md(r'''
---

## Reading this

- **Gap improves / turns clearly positive** -- capacity is the lever that
  closes HOLO4K's remaining gap, same story as COACH420 and CryptoBench;
  worth extending to 8 seeds.
- **Gap holds flat (within noise)** -- capacity doesn't move HOLO4K either;
  combined with a flat Phase93 result this would suggest round-3
  scale+topology is genuinely at a ceiling for this architecture on these
  benchmarks, and the ESM2-capacity lever is exhausted.
- **Gap worsens** -- worth scrutiny (fp16 precision, truncation at the
  1000-residue cap interacting differently with the wider embedding)
  before accepting.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase94_esm3b_holo4k.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
