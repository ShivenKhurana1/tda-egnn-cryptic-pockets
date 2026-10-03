import torch
from pathlib import Path
from collections import Counter

GRAPH = Path("phase159_input/phase159_cryptobench_anmmsf_graphs.pt")

print("=" * 110)
print("PHASE 182A - CRYPTOBENCH EXPERIMENTAL POCKET LABEL AUDIT")
print("=" * 110)

obj = torch.load(GRAPH, map_location="cpu", weights_only=False)

print("\nTop-level object type:")
print(type(obj))

if isinstance(obj, dict):
    print("\nTop-level keys:")
    for k, v in obj.items():
        print(f"  {k!r}: {type(v)}", end="")
        try:
            print(f", len={len(v)}")
        except Exception:
            print()
else:
    print("Unexpected top-level object.")
    raise SystemExit

dataset = obj.get("dataset", obj)

print("\nDataset type:")
print(type(dataset))

try:
    print("Dataset length:", len(dataset))
except Exception:
    pass

# Inspect the first few graph records
if isinstance(dataset, dict):
    items = list(dataset.items())
else:
    items = list(enumerate(dataset))

print("\n" + "=" * 110)
print("GRAPH RECORD STRUCTURE")
print("=" * 110)

for idx, (key, graph) in enumerate(items[:5]):
    print(f"\n--- RECORD {idx} ---")
    print("Dictionary key:", key)
    print("Graph type:", type(graph))

    if hasattr(graph, "__dict__"):
        attrs = vars(graph)
        print("Attributes:")
        for name, value in attrs.items():
            shape = getattr(value, "shape", None)
            dtype = getattr(value, "dtype", None)
            if shape is not None:
                print(f"  {name}: shape={shape}, dtype={dtype}")
            else:
                print(f"  {name}: type={type(value)}, value={repr(value)[:300]}")

    elif isinstance(graph, dict):
        print("Dictionary keys:")
        for name, value in graph.items():
            shape = getattr(value, "shape", None)
            dtype = getattr(value, "dtype", None)
            if shape is not None:
                print(f"  {name}: shape={shape}, dtype={dtype}")
            else:
                print(f"  {name}: type={type(value)}, value={repr(value)[:300]}")

    else:
        print(repr(graph)[:2000])

# Search recursively for potentially relevant label/ligand/pocket fields
print("\n" + "=" * 110)
print("SEARCHING FOR POTENTIAL POCKET / LIGAND / LABEL FIELDS")
print("=" * 110)

KEYWORDS = [
    "pocket",
    "ligand",
    "binding",
    "site",
    "label",
    "target",
    "residue",
    "contact",
    "holo",
    "apo",
    "center",
    "true",
    "positive",
]

found = []

def inspect_object(obj, path="root", depth=0, max_depth=4):
    if depth > max_depth:
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            keystr = str(k).lower()
            if any(word in keystr for word in KEYWORDS):
                found.append((path + "." + str(k), type(v), getattr(v, "shape", None), repr(v)[:500]))
            inspect_object(v, path + "." + str(k), depth + 1, max_depth)

    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj[:10]):
            inspect_object(v, f"{path}[{i}]", depth + 1, max_depth)

    elif hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            keystr = str(k).lower()
            if any(word in keystr for word in KEYWORDS):
                found.append((path + "." + str(k), type(v), getattr(v, "shape", None), repr(v)[:500]))
            inspect_object(v, path + "." + str(k), depth + 1, max_depth)

inspect_object(obj)

# Deduplicate
seen = set()
unique_found = []

for item in found:
    identifier = item[0]
    if identifier not in seen:
        seen.add(identifier)
        unique_found.append(item)

for path, typ, shape, preview in unique_found:
    print(f"\n{path}")
    print(f"  type: {typ}")
    if shape is not None:
        print(f"  shape: {shape}")
    print(f"  preview: {preview}")

print("\n" + "=" * 110)
print("SUMMARY")
print("=" * 110)
print("Potentially relevant fields found:", len(unique_found))

if unique_found:
    print("\nFields containing relevant keywords were found above.")
    print("We will use the actual field structure in Phase 182B.")
else:
    print("\nNo explicit pocket/ligand/label fields were found.")
    print("If this is the case, we will inspect the original CryptoBench metadata next.")

print("\nDone.")
