import torch
import numpy as np
from pathlib import Path
from collections import Counter

GRAPH = Path("phase159_input/phase159_cryptobench_anmmsf_graphs.pt")

print("=" * 110)
print("PHASE 182B - NODE LABEL (y) SEMANTICS AUDIT")
print("=" * 110)

obj = torch.load(GRAPH, map_location="cpu", weights_only=False)
dataset = obj["dataset"]

print(f"\nNumber of graphs: {len(dataset)}")

# ---------------------------------------------------------------------
# 1. Inspect every attribute in representative graphs
# ---------------------------------------------------------------------
print("\n" + "=" * 110)
print("FULL ATTRIBUTE INVENTORY")
print("=" * 110)

for i in range(min(10, len(dataset))):
    g = dataset[i]

    print(f"\n--- GRAPH {i} ---")

    if hasattr(g, "_store"):
        mapping = g._store._mapping
        for name, value in mapping.items():
            shape = getattr(value, "shape", None)
            dtype = getattr(value, "dtype", None)

            if torch.is_tensor(value):
                flat = value.detach().cpu().reshape(-1)
                preview = flat[:10].tolist()

                if value.numel() > 0:
                    try:
                        vmin = float(flat.min())
                        vmax = float(flat.max())
                    except Exception:
                        vmin = vmax = None
                else:
                    vmin = vmax = None

                print(
                    f"{name:25s} "
                    f"shape={tuple(value.shape)!s:18s} "
                    f"dtype={str(dtype):12s} "
                    f"min={vmin!s:12s} "
                    f"max={vmax!s:12s} "
                    f"preview={preview}"
                )
            else:
                print(
                    f"{name:25s} "
                    f"type={type(value)} "
                    f"value={repr(value)[:250]}"
                )

# ---------------------------------------------------------------------
# 2. Detailed investigation of y
# ---------------------------------------------------------------------
print("\n" + "=" * 110)
print("DETAILED y ANALYSIS")
print("=" * 110)

all_values = []
graph_summaries = []

for i, g in enumerate(dataset):
    if not hasattr(g, "y"):
        graph_summaries.append({
            "graph": i,
            "has_y": False,
        })
        continue

    y = g.y.detach().cpu()

    flat = y.reshape(-1).numpy()

    unique, counts = np.unique(flat, return_counts=True)

    graph_summaries.append({
        "graph": i,
        "has_y": True,
        "shape": tuple(y.shape),
        "dtype": str(y.dtype),
        "n_values": len(flat),
        "n_unique": len(unique),
        "unique_values": unique[:20].tolist(),
        "counts": counts[:20].tolist(),
        "positive_count_if_binary": int(np.sum(flat > 0)),
        "positive_fraction": float(np.mean(flat > 0)),
        "mean": float(np.mean(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    })

    all_values.extend(flat.tolist())

for row in graph_summaries[:20]:
    print(row)

# ---------------------------------------------------------------------
# 3. Global y distribution
# ---------------------------------------------------------------------
print("\n" + "=" * 110)
print("GLOBAL y DISTRIBUTION")
print("=" * 110)

all_values = np.asarray(all_values)

print("Total y values:", len(all_values))
print("dtype:", all_values.dtype)

unique, counts = np.unique(all_values, return_counts=True)

print("\nUnique values:")
for u, c in zip(unique[:100], counts[:100]):
    print(f"  {u!r}: {c:,} ({c / len(all_values):.6f})")

if len(unique) > 100:
    print(f"... {len(unique)-100} additional unique values omitted")

# ---------------------------------------------------------------------
# 4. Determine whether y looks binary
# ---------------------------------------------------------------------
print("\n" + "=" * 110)
print("BINARY-LABEL CHECK")
print("=" * 110)

is_binary = np.all(np.isin(unique, [0, 1]))
is_binary_pm = np.all(np.isin(unique, [-1, 1]))
is_binary_bool = np.all(np.isin(unique, [False, True]))

print("Values subset of {0,1}:", is_binary)
print("Values subset of {-1,1}:", is_binary_pm)
print("Boolean-like:", is_binary_bool)

if is_binary:
    print("Positive fraction:", float(np.mean(all_values == 1)))
    print("Negative/zero fraction:", float(np.mean(all_values == 0)))

# ---------------------------------------------------------------------
# 5. Per-graph positive-label statistics
# ---------------------------------------------------------------------
print("\n" + "=" * 110)
print("PER-GRAPH POSITIVE-LABEL STATISTICS")
print("=" * 110)

positive_fracs = []
positive_counts = []

for i, g in enumerate(dataset):
    if not hasattr(g, "y"):
        continue

    y = g.y.detach().cpu().reshape(-1).numpy()

    if len(y) == 0:
        continue

    npos = int(np.sum(y > 0))
    frac = float(np.mean(y > 0))

    positive_counts.append(npos)
    positive_fracs.append(frac)

    if i < 30:
        pdb = getattr(g, "pdb_id", "UNKNOWN")
        chain = getattr(g, "chain_id", "UNKNOWN")

        if torch.is_tensor(pdb):
            pdb = pdb.tolist()
        if torch.is_tensor(chain):
            chain = chain.tolist()

        print(
            f"graph={i:3d} "
            f"pdb={pdb!s:10s} "
            f"chain={chain!s:5s} "
            f"nodes={len(y):5d} "
            f"positive={npos:4d} "
            f"fraction={frac:.6f}"
        )

print("\nAcross graphs:")
print("  graphs with y:", len(positive_fracs))
print("  mean positive fraction:", np.mean(positive_fracs))
print("  median positive fraction:", np.median(positive_fracs))
print("  min positive fraction:", np.min(positive_fracs))
print("  max positive fraction:", np.max(positive_fracs))
print("  mean positive count:", np.mean(positive_counts))
print("  median positive count:", np.median(positive_counts))

# ---------------------------------------------------------------------
# 6. Check whether y aligns exactly with graph nodes
# ---------------------------------------------------------------------
print("\n" + "=" * 110)
print("NODE ALIGNMENT CHECK")
print("=" * 110)

alignment_ok = True

for i, g in enumerate(dataset):
    if not hasattr(g, "y") or not hasattr(g, "x"):
        print(f"Graph {i}: missing x or y")
        alignment_ok = False
        continue

    if g.y.shape[0] != g.x.shape[0]:
        print(
            f"Graph {i}: y.shape={tuple(g.y.shape)}, "
            f"x.shape={tuple(g.x.shape)}"
        )
        alignment_ok = False

print("y first dimension equals number of nodes for all graphs:", alignment_ok)

# ---------------------------------------------------------------------
# 7. Compare y with true_center proximity
#
# This is NOT being used to define labels. It is only a diagnostic:
# if y is a pocket label, positive nodes should generally lie closer
# to the experimental center than negative nodes.
# ---------------------------------------------------------------------
print("\n" + "=" * 110)
print("y vs TRUE-CENTER DIAGNOSTIC")
print("=" * 110)

positive_distances = []
negative_distances = []

for i, g in enumerate(dataset):
    if not all(hasattr(g, attr) for attr in ["y", "pos", "true_center"]):
        continue

    y = g.y.detach().cpu().reshape(-1).numpy()
    pos = g.pos.detach().cpu().numpy()
    center = g.true_center.detach().cpu().numpy().reshape(3)

    if len(y) != len(pos):
        continue

    distances = np.linalg.norm(pos - center[None, :], axis=1)

    positive_distances.extend(distances[y > 0].tolist())
    negative_distances.extend(distances[y <= 0].tolist())

print("Positive nodes:", len(positive_distances))
print("Non-positive nodes:", len(negative_distances))

if positive_distances:
    print("\nPositive-node distance to true_center:")
    print("  mean:", np.mean(positive_distances))
    print("  median:", np.median(positive_distances))
    print("  25th percentile:", np.percentile(positive_distances, 25))
    print("  75th percentile:", np.percentile(positive_distances, 75))

if negative_distances:
    print("\nNon-positive-node distance to true_center:")
    print("  mean:", np.mean(negative_distances))
    print("  median:", np.median(negative_distances))
    print("  25th percentile:", np.percentile(negative_distances, 25))
    print("  75th percentile:", np.percentile(negative_distances, 75))

# ---------------------------------------------------------------------
# 8. Print representative positive node indices
# ---------------------------------------------------------------------
print("\n" + "=" * 110)
print("REPRESENTATIVE POSITIVE NODE INDICES")
print("=" * 110)

for i in range(min(10, len(dataset))):
    g = dataset[i]

    if not hasattr(g, "y"):
        continue

    y = g.y.detach().cpu().reshape(-1).numpy()
    positive_idx = np.where(y > 0)[0]

    print(
        f"graph={i}: "
        f"positive indices ({len(positive_idx)}): "
        f"{positive_idx[:30].tolist()}"
    )

print("\n" + "=" * 110)
print("INTERPRETATION")
print("=" * 110)

if is_binary:
    print(
        "y is binary-valued. The next step is to determine whether these "
        "binary node labels represent experimental pocket residues."
    )
else:
    print(
        "y is not a simple binary {0,1} label. We must determine its "
        "semantic meaning before using it as a pocket-residue target."
    )

print("\nNo model training or prediction was performed.")
print("This phase only audits the existing labels.")
