import torch
from pathlib import Path
import os
import glob

GRAPH = Path("phase159_input/phase159_cryptobench_anmmsf_graphs.pt")

print("=" * 110)
print("PHASE 182C - NODE-LABEL PROVENANCE AND CRYPTOBENCH SOURCE AUDIT")
print("=" * 110)

obj = torch.load(GRAPH, map_location="cpu", weights_only=False)

print("\n" + "=" * 110)
print("GRAPH METADATA")
print("=" * 110)

for key in [
    "source",
    "aa_alphabet",
    "topo_names",
    "burial_names",
    "dropped_names",
    "esm_dim",
    "anm_cutoff",
    "anm_n_modes",
]:
    value = obj.get(key, "<NOT PRESENT>")
    print(f"\n{key}:")
    print(value)

# ---------------------------------------------------------------------
# Search local project files for the construction of y.
# ---------------------------------------------------------------------
print("\n" + "=" * 110)
print("SEARCHING PROJECT FILES FOR y-LABEL CONSTRUCTION")
print("=" * 110)

ROOT = Path(".")

patterns = [
    "true_center",
    "y =",
    "y=",
    "data.y",
    "edge_label",
    "pocket",
    "ligand",
    "cryptobench",
    "binding",
    "residue",
]

files = []

for ext in ["*.py", "*.ipynb", "*.md", "*.txt", "*.sh"]:
    files.extend(ROOT.rglob(ext))

# Exclude giant/generated directories where possible
files = [
    p for p in files
    if not any(part in {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
    } for part in p.parts)
]

print("Candidate local files:", len(files))

matches = []

for path in files:
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        continue

    lines = text.splitlines()

    for lineno, line in enumerate(lines, start=1):
        lower = line.lower()

        if any(pattern.lower() in lower for pattern in patterns):
            matches.append((path, lineno, line.strip()))

print("Matching lines:", len(matches))

for path, lineno, line in matches[:500]:
    print(f"{path}:{lineno}: {line}")

# ---------------------------------------------------------------------
# Search specifically for likely graph-generation scripts.
# ---------------------------------------------------------------------
print("\n" + "=" * 110)
print("LIKELY GRAPH-GENERATION FILES")
print("=" * 110)

keywords = [
    "graph",
    "cryptobench",
    "dataset",
    "pocket",
    "anm",
    "topo",
]

scores = []

for path in files:
    name = path.name.lower()
    score = sum(k in name for k in keywords)

    if score > 0:
        scores.append((score, str(path)))

for score, path in sorted(scores, reverse=True)[:100]:
    print(f"{score:2d}  {path}")

# ---------------------------------------------------------------------
# Inspect graph-level train/val/test counts.
# ---------------------------------------------------------------------
print("\n" + "=" * 110)
print("SPLIT COUNTS")
print("=" * 110)

split_counts = {}

for g in obj["dataset"]:
    split = getattr(g, "split", "UNKNOWN")
    split_counts[split] = split_counts.get(split, 0) + 1

print(split_counts)

# ---------------------------------------------------------------------
# Check whether all graphs have y and true_center.
# ---------------------------------------------------------------------
print("\n" + "=" * 110)
print("FIELD COMPLETENESS")
print("=" * 110)

fields = [
    "x",
    "edge_index",
    "edge_attr",
    "y",
    "pos",
    "true_center",
    "pdb_id",
    "chain_id",
    "split",
]

for field in fields:
    count = sum(hasattr(g, field) for g in obj["dataset"])
    print(f"{field:20s}: {count}/{len(obj['dataset'])}")

print("\n" + "=" * 110)
print("NEXT STEP")
print("=" * 110)
print(
    "Use the local source/code matches above to establish exactly how "
    "the binary y labels were generated. Do not redefine y from distances."
)
