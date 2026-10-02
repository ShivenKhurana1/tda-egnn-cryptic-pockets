import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = "/Users/shivenkhurana/bioproj/cryptic-pocket"
P = torch.load("/tmp/p266c_fresh/phase266_full_preds.pt", map_location="cpu", weights_only=False)
seeds = sorted(P["seeds"])

labels = json.load(open(f"{ROOT}/phase276/phase276a_crypticity_labels.json"))
label_of = {tuple(k.rsplit("_", 1)): v["label"] for k, v in labels.items()}

def ensemble(split):
    S0 = P["seeds"][seeds[0]][split]
    out = {}
    for k in S0:
        out[k] = dict(
            p=np.mean([P["seeds"][s][split][k]["p"] for s in seeds], axis=0),
            y=S0[k]["y"],
            c=np.mean([P["seeds"][s][split][k]["c"] for s in seeds], axis=0),
            tc=S0[k]["tc"],
        )
    return out

records = []
for split in ("val", "test"):
    ens = ensemble(split)
    for k, v in ens.items():
        p, y = v["p"], v["y"].astype(int)
        n = len(y); npos = int(y.sum())
        err = float(np.linalg.norm(v["c"] - v["tc"].reshape(-1)))
        rec = dict(
            key=f"{k[0]}_{k[1]}", pdb=k[0], chain=k[1], split=split,
            n=n, npos=npos, pct_pos=round(100*npos/n, 2),
            dcc_err=round(err, 2), dcc4=bool(err <= 4), dcc8=bool(err <= 8),
            crypticity=label_of.get((k[0], k[1])),
            p=[round(float(x), 3) for x in p],
            y=[int(x) for x in y],
        )
        if 0 < npos < n:
            rec["roc"] = round(float(roc_auc_score(y, p)), 4)
            rec["pr"] = round(float(average_precision_score(y, p)), 4)
        else:
            rec["roc"] = None; rec["pr"] = None
        records.append(rec)

print(f"{len(records)} chains total ({sum(1 for r in records if r['split']=='test')} test, "
      f"{sum(1 for r in records if r['split']=='val')} val)")
print("crypticity counts:", {})
from collections import Counter
print(Counter(r["crypticity"] for r in records if r["split"]=="test"))

# summary stats for the header
test_recs = [r for r in records if r["split"] == "test"]
all_p = np.concatenate([np.array(r["p"]) for r in test_recs])
all_y = np.concatenate([np.array(r["y"]) for r in test_recs])
summary = dict(
    n_test_chains=len(test_recs),
    n_val_chains=sum(1 for r in records if r["split"]=="val"),
    n_test_residues=int(len(all_y)),
    pooled_roc=round(float(roc_auc_score(all_y, all_p)), 4),
    pooled_pr=round(float(average_precision_score(all_y, all_p)), 4),
    pooled_pos_rate=round(100*float(all_y.mean()), 2),
    n_cryptic=sum(1 for r in test_recs if r["crypticity"]=="cryptic"),
    n_preformed=sum(1 for r in test_recs if r["crypticity"]=="preformed"),
    dcc4_rate=round(100*sum(r["dcc4"] for r in test_recs)/len(test_recs), 1),
)
print(summary)

out = dict(summary=summary, records=records)
json.dump(out, open("data.json", "w"), separators=(",", ":"))
import os
print("file size MB:", os.path.getsize("data.json")/1e6)
