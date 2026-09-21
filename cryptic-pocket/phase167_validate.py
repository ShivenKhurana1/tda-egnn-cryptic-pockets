
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax as pyg_softmax


ROOT = Path.home() / "bioproj" / "cryptic-pocket"

GRAPH_FILE = ROOT / "phase159_input/phase159_cryptobench_anmmsf_graphs.pt"
MAPPING_FILE = ROOT / "cryptobench_binding_mapping.csv"
EXPERIMENTAL_FILE = ROOT / "phase167_experimental_sites.csv"

MODEL_FILES = {
    "esm_structure": sorted(
        (ROOT / "phase166A/output/cache").glob(
            "phase166_esm_structure_seed*.pt"
        )
    ),
    "full_tda": sorted(
        (ROOT / "phase166A/phase166B/output/cache").glob(
            "phase166_full_tda_seed*.pt"
        )
    ),
    "full_model": sorted(
        (ROOT / "phase166A/phase166B/phase166C/output/cache").glob(
            "phase166_full_model_seed*.pt"
        )
    ),
}

OUTPUT_DIR = ROOT / "phase167_validation"
OUTPUT_DIR.mkdir(exist_ok=True)

N_PRE = 508
TOPO_DIM = 34


# ============================================================
# Exact Phase 166 architecture
# ============================================================

class EGCL(nn.Module):

    def __init__(self, h, ea=2, act=nn.SiLU, uc=True):
        super().__init__()
        self.uc = uc

        self.e = nn.Sequential(
            nn.Linear(2*h + 1 + ea, h),
            act(),
            nn.Linear(h, h),
            act()
        )

        self.n = nn.Sequential(
            nn.Linear(2*h, h),
            act(),
            nn.Linear(h, h)
        )

        self.a = nn.Sequential(
            nn.Linear(h, 1),
            nn.Sigmoid()
        )

        if uc:
            self.c = nn.Sequential(
                nn.Linear(h, h),
                act(),
                nn.Linear(h, 1, bias=False)
            )
            nn.init.xavier_uniform_(
                self.c[-1].weight,
                gain=0.001
            )

        self.norm = nn.LayerNorm(h)

    def forward(self, h, x, ei, ea):

        src, dst = ei

        rel = x[src] - x[dst]

        d2 = (rel**2).sum(
            -1,
            keepdim=True
        )

        m = self.e(
            torch.cat(
                [h[src], h[dst], d2, ea],
                -1
            )
        )

        m = m * self.a(m)

        agg = torch.zeros_like(h).index_add_(
            0,
            dst,
            m
        )

        deg = torch.zeros(
            h.size(0),
            1,
            device=h.device
        ).index_add_(
            0,
            dst,
            torch.ones_like(d2)
        ).clamp(min=1)

        h = self.norm(
            h + self.n(
                torch.cat(
                    [h, agg / deg],
                    -1
                )
            )
        )

        if self.uc:

            w = (
                self.c(m)
                * rel
                / (d2.sqrt() + 1.0)
            )

            x = (
                x
                + torch.zeros_like(x).index_add_(
                    0,
                    dst,
                    w
                ) / deg
            )

        return h, x


class Fus(nn.Module):

    def __init__(self, h, t):
        super().__init__()

        self.p = nn.Linear(t, h)

        self.g = nn.Sequential(
            nn.Linear(h+t, h),
            nn.Sigmoid()
        )

    def forward(self, h, t):

        return (
            h
            + self.g(
                torch.cat([h, t], -1)
            ) * self.p(t)
        )


class LocalizingEGNN(nn.Module):

    def __init__(
        self,
        n_pre,
        td,
        h=128,
        L=4,
        dr=0.1
    ):

        super().__init__()

        self.n_pre = n_pre

        self.embed = nn.Sequential(
            nn.Linear(n_pre + td, h),
            nn.SiLU(),
            nn.Linear(h, h)
        )

        self.ls = nn.ModuleList([
            EGCL(
                h,
                uc=(i < L-1)
            )
            for i in range(L)
        ])

        self.fs = nn.ModuleList([
            Fus(h, td)
            for _ in range(L)
        ])

        self.head = nn.Sequential(
            nn.Linear(h, h//2),
            nn.SiLU(),
            nn.Dropout(dr),
            nn.Linear(h//2, 1)
        )

        self.loc_head = nn.Sequential(
            nn.Linear(h, h//2),
            nn.SiLU(),
            nn.Linear(h//2, 1)
        )

    def forward(self, d):

        h = self.embed(d.x)
        orig_pos = d.pos.clone()

        pos = d.pos

        t = d.x[:, self.n_pre:]

        for l, f in zip(self.ls, self.fs):

            h, pos = l(
                h,
                pos,
                d.edge_index,
                d.edge_attr
            )

            h = f(h, t)

        logits = self.head(h).squeeze(-1)

        batch = (
            d.batch
            if getattr(d, "batch", None) is not None
            else torch.zeros(
                h.size(0),
                dtype=torch.long,
                device=h.device
            )
        )

        loc_logit = self.loc_head(h).squeeze(-1)

        attn = pyg_softmax(
            loc_logit,
            batch
        )

        pred_center = torch.zeros(
            int(batch.max().item()) + 1,
            3,
            device=h.device
        ).index_add_(
            0,
            batch,
            attn.unsqueeze(-1) * orig_pos
        )

        return logits, pred_center


# ============================================================
# Load graph dataset
# ============================================================

print("Loading graph dataset...")

blob = torch.load(
    GRAPH_FILE,
    map_location="cpu",
    weights_only=False
)

dataset = blob["dataset"]

print("Graphs:", len(dataset))
print("Node dimensions:", dataset[0].x.shape[1])


# ============================================================
# Reconstruct exact Phase 166 hard split
# ============================================================

manifest_path = (
    ROOT
    / "phase166A/output/phase166_hard_split_manifest.json"
)

manifest = json.load(open(manifest_path))

for d in dataset:

    emb = d.x[:, 27:507].float().mean(0).numpy()

# Use manifest assignments directly.
# Cluster labels must be reconstructed exactly.

from sklearn.cluster import KMeans

emb = np.stack([
    d.x[:,27:507].float().mean(0).numpy()
    for d in dataset
])

emb = emb / (
    np.linalg.norm(
        emb,
        axis=1,
        keepdims=True
    ) + 1e-12
)

km = KMeans(
    n_clusters=30,
    random_state=166,
    n_init=20,
    max_iter=300
)

labels = km.fit_predict(emb)

cluster_assignment = {
    int(k): v
    for k, v in manifest["cluster_assignment"].items()
}

hard_split = {
    i: cluster_assignment[int(labels[i])]
    for i in range(len(dataset))
}

for i, d in enumerate(dataset):
    d.hard_split = hard_split[i]


test_indices = [
    i for i, d in enumerate(dataset)
    if d.hard_split == "test_hard"
]

test_graphs = [
    dataset[i]
    for i in test_indices
]

print("Hard test graphs:", len(test_graphs))


# ============================================================
# Mapping / experimental data
# ============================================================

mapping = pd.read_csv(MAPPING_FILE)

experimental = pd.read_csv(EXPERIMENTAL_FILE)

experimental = experimental.set_index(
    ["apo_pdb_id", "apo_chain"]
)

mapping = mapping.set_index(
    ["apo_pdb", "apo_chain"]
)


# ============================================================
# Build apo residue identifiers
# ============================================================

def graph_residue_keys(d):

    keys = []

    # CryptoBench graph objects generally retain residue
    # identifiers in the graph metadata. Try common forms.

    if hasattr(d, "residue_ids"):
        vals = d.residue_ids

        if torch.is_tensor(vals):
            vals = vals.tolist()

        keys = [
            str(x)
            for x in vals
        ]

    elif hasattr(d, "res_ids"):

        vals = d.res_ids

        if torch.is_tensor(vals):
            vals = vals.tolist()

        keys = [
            str(x)
            for x in vals
        ]

    elif hasattr(d, "resseq"):

        vals = d.resseq

        if torch.is_tensor(vals):
            vals = vals.tolist()

        keys = [
            str(int(x))
            for x in vals
        ]

    return keys


# ============================================================
# Convert CryptoBench selection strings to residue numbers
# ============================================================

def parse_selection(selection):

    if pd.isna(selection):
        return set()

    selection = str(selection)

    result = set()

    for item in selection.replace(",", " ").split():

        item = item.strip()

        if not item:
            continue

        # Expected forms include things such as B_12,
        # and possibly comma/space separated selections.

        if "_" in item:

            chain, number = item.split("_", 1)

            try:
                result.add(
                    (chain, int(number))
                )
            except ValueError:
                pass

        else:

            # Try bare residue numbers.
            try:
                result.add(
                    (None, int(item))
                )
            except ValueError:
                pass

    return result


# ============================================================
# Model masking
# ============================================================

ARMS = {
    "esm_structure": [],
    "full_tda": list(range(508,542)),
    "full_model": list(range(508,542)) + [507],
}


def masked_graph(d, arm):

    d2 = copy.copy(d)

    d2.x = d.x.clone()

    keep = torch.zeros(
        d2.x.size(1),
        dtype=torch.bool
    )

    keep[0:27] = True
    keep[27:507] = True

    keep[ARMS[arm]] = True

    d2.x[:, ~keep] = 0.0

    return d2


# ============================================================
# Predict one model
# ============================================================

DEVICE = torch.device("cpu")


def predict(model_path, arm):

    model = LocalizingEGNN(
        N_PRE,
        TOPO_DIM
    )

    state = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state)
    model.eval()

    graphs = [
        masked_graph(d, arm)
        for d in test_graphs
    ]

    loader = DataLoader(
        graphs,
        batch_size=8,
        shuffle=False
    )

    rows = []
    offset = 0

    with torch.no_grad():

        for batch in loader:

            logits, pred_center = model(batch)

            probs = torch.sigmoid(logits)

            # Number of graphs in this batch
            n_graphs = int(batch.num_graphs)

            for bidx in range(n_graphs):

                graph = graphs[offset + bidx]

                node_mask = batch.batch == bidx
                p = probs[node_mask].cpu().numpy()

                center = pred_center[bidx].cpu().numpy()

                for j, prob in enumerate(p):

                    rows.append({
                        "apo_pdb_id": str(
                            graph.pdb_id
                        ).upper(),

                        "apo_chain": str(
                            graph.chain_id
                        ),

                        "node_index": j,

                        "probability": float(
                            prob
                        ),

                        "predicted_center_x": float(
                            center[0]
                        ),

                        "predicted_center_y": float(
                            center[1]
                        ),

                        "predicted_center_z": float(
                            center[2]
                        ),

                        "arm": arm,

                        "seed": int(
                            Path(model_path).stem
                            .split("seed")[-1]
                        )
                    })

            # Advance once per batch, not once per graph/node.
            offset += n_graphs

    assert offset == len(graphs), (
        f"Prediction offset mismatch: "
        f"{offset} != {len(graphs)}"
    )

    return pd.DataFrame(rows)


# ============================================================
# Run all 24 models
# ============================================================

all_predictions = []

for arm, files in MODEL_FILES.items():

    print()
    print("="*70)
    print(
        arm,
        "models:",
        len(files)
    )
    print("="*70)

    for model_path in files:

        print(
            "Predicting:",
            model_path.name
        )

        pred = predict(
            model_path,
            arm
        )

        all_predictions.append(pred)

predictions = pd.concat(
    all_predictions,
    ignore_index=True
)

prediction_file = (
    OUTPUT_DIR
    / "phase167_residue_predictions.csv"
)

predictions.to_csv(
    prediction_file,
    index=False
)

print()
print("Saved predictions:")
print(prediction_file)

print()
print(
    "Prediction rows:",
    len(predictions)
)

print(
    "Proteins:",
    predictions[
        ["apo_pdb_id","apo_chain"]
    ].drop_duplicates().shape[0]
)

print()
print("DONE")
