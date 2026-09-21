#!/usr/bin/env python3
"""Phase 148: CryptoBench weighted cross-scale ensemble, FINAL version --
650M-transfer (Phase146, test AUC 0.9141/MCC 0.4473) replaces 650M-
scratch (Phase47, test AUC 0.8873/MCC 0.3986) as the 650M arm. Now all
3 scales are transfer-learned. Grid-searches ensemble weights on VAL
ONLY (never touches test until the final evaluation), same discipline
as Phase131."""
import json
from pathlib import Path

nb130 = json.load(open("/Users/shivenkhurana/bioproj/phase130_cryptobench_combined.ipynb"))
c130 = nb130["cells"]
s130 = lambda i: c130[i]["source"]

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 148 — CryptoBench: Weighted Cross-Scale Ensemble, All-Transfer

Phase131's weighted ensemble used 650M-scratch (test AUC 0.8873, MCC
0.3986) as the weakest arm. Phase146/147 replaced it with 650M-transfer
(test AUC 0.9141, MCC 0.4473, now the middle-strength arm, not the
weakest). All 3 scales are now transfer-learned from main-pipeline
COACH420 checkpoints. Grid-searches ensemble weights on VAL ONLY,
applies the winning weights to TEST once.
''')

md(r'''## Cell 1 — Install dependencies''')
code(s130(2))

md(r'''## Cell 2 — Model + setup (identical to Phase130/131)''')
code(s130(4))

md(r'''## Cell 3 — Load each scale's blob + 8 checkpoints (ALL 3 scales now transfer-learned), predict on VAL and TEST''')
code(r'''
SCALES = {
    "35M":  dict(blob="phase42_cryptobench_graphs.pt",         ckpt_prefix="cryptobench_transfer_seed"),
    "650M": dict(blob="phase45_cryptobench_graphs_esm650m.pt", ckpt_prefix="cryptobench_esm650m_transfer_seed"),
    "3B":   dict(blob="phase48_cryptobench_graphs_esm3b.pt",   ckpt_prefix="cryptobench_esm3b_transfer_seed"),
}

scale_val_probs, scale_val_y, scale_val_keys = {}, {}, {}
scale_test_probs, scale_test_y, scale_test_keys = {}, {}, {}
t0 = time.time()
for scale, cfg in SCALES.items():
    blob = torch.load(locate(cfg["blob"]), map_location="cpu", weights_only=False)
    dataset = blob["dataset"]
    N_PRE = int(blob["n_pre_topo"]); TOPO_DIM = int(blob["topo_dim"])
    val_set = [d for d in dataset if d.split == "val"]
    test_set = [d for d in dataset if d.split == "test"]
    val_loader = DataLoader(val_set, batch_size=16)
    test_loader = DataLoader(test_set, batch_size=16)
    scale_val_keys[scale] = [(d.pdb_id, d.chain_id, int(d.num_nodes)) for d in val_set]
    scale_test_keys[scale] = [(d.pdb_id, d.chain_id, int(d.num_nodes)) for d in test_set]

    val_seed_probs, test_seed_probs = [], []
    for seed in range(8):
        ckpt = locate(f"{cfg['ckpt_prefix']}{seed}.pt")
        m = LocalizingEGNN(N_PRE, TOPO_DIM).to(DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE)); m.eval()
        vp, vy = predict_seed(m, val_loader)
        tp, ty = predict_seed(m, test_loader)
        val_seed_probs.append(vp); test_seed_probs.append(tp)
        if seed == 0:
            scale_val_y[scale] = vy; scale_test_y[scale] = ty
    scale_val_probs[scale] = np.mean(val_seed_probs, axis=0)
    scale_test_probs[scale] = np.mean(test_seed_probs, axis=0)
    print(f"  {scale:<6} 8-seed val+test predictions ready ({cfg['ckpt_prefix']}*)  [{(time.time()-t0)/60:.1f}m]")

for split_name, keys_dict in [("val", scale_val_keys), ("test", scale_test_keys)]:
    ref = keys_dict["35M"]
    for scale in ("650M", "3B"):
        assert keys_dict[scale] == ref, f"{split_name}: {scale} key order mismatch vs 35M"
print(f"\nalignment verified on both val and test across all 3 scales")
''')

md(r'''## Cell 4 — Grid-search ensemble weights on VAL ONLY, apply winning weights + val-tuned threshold to TEST''')
code(r'''
def metrics_at(p, y, thresh):
    pred = (p >= thresh).astype(int)
    return {"auc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
            "mcc": matthews_corrcoef(y, pred), "f1": f1_score(y, pred)}

def best_thresh(p, y):
    best_t, best_mcc = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        mcc = matthews_corrcoef(y, (p >= t).astype(int))
        if mcc > best_mcc: best_mcc, best_t = mcc, t
    return best_t, best_mcc

y_val_ref = scale_val_y["35M"]
y_test_ref = scale_test_y["35M"]
SCALE_NAMES = ["35M", "650M", "3B"]

# individual-scale TEST metrics for reference (each scale alone, val-tuned threshold)
print(f"{'scale (alone)':<28}{'AUC':>8}{'AUPRC':>8}{'MCC':>8}{'F1':>8}")
print("-"*60)
solo_results = {}
for scale in SCALE_NAMES:
    t, _ = best_thresh(scale_val_probs[scale], y_val_ref)
    r = metrics_at(scale_test_probs[scale], y_test_ref, t)
    solo_results[scale] = r
    print(f"{scale:<28}{r['auc']:>8.4f}{r['auprc']:>8.4f}{r['mcc']:>8.4f}{r['f1']:>8.4f}")

# equal-weight ensemble (1/3 each), for reference
equal_p_val = sum(scale_val_probs[s] for s in SCALE_NAMES) / 3.0
equal_p_test = sum(scale_test_probs[s] for s in SCALE_NAMES) / 3.0
t_eq, _ = best_thresh(equal_p_val, y_val_ref)
equal_result = metrics_at(equal_p_test, y_test_ref, t_eq)

# grid search: weights on a 0.1 simplex (w1+w2+w3=1, each a multiple of 0.1)
best_w, best_val_mcc, best_val_auc_at_w = None, -1.0, None
grid_results = []
for w1 in np.arange(0.0, 1.01, 0.1):
    for w2 in np.arange(0.0, 1.01 - w1, 0.1):
        w3 = 1.0 - w1 - w2
        if w3 < -1e-9: continue
        w3 = max(w3, 0.0)
        weights = {"35M": w1, "650M": w2, "3B": w3}
        val_p = sum(weights[s] * scale_val_probs[s] for s in SCALE_NAMES)
        t, val_mcc = best_thresh(val_p, y_val_ref)
        val_auc = roc_auc_score(y_val_ref, val_p)
        grid_results.append(dict(w35m=w1, w650m=w2, w3b=w3, val_mcc=val_mcc, val_auc=val_auc, thresh=t))
        if val_mcc > best_val_mcc:
            best_val_mcc, best_w, best_val_auc_at_w = val_mcc, weights, val_auc

print(f"\nbest weights by VAL MCC: 35M={best_w['35M']:.1f}  650M={best_w['650M']:.1f}  3B={best_w['3B']:.1f}")
print(f"  (val MCC {best_val_mcc:.4f}, val AUC {best_val_auc_at_w:.4f})")

test_p_weighted = sum(best_w[s] * scale_test_probs[s] for s in SCALE_NAMES)
val_p_weighted = sum(best_w[s] * scale_val_probs[s] for s in SCALE_NAMES)
final_thresh, _ = best_thresh(val_p_weighted, y_val_ref)
weighted_result = metrics_at(test_p_weighted, y_test_ref, final_thresh)

best_w_auc, best_val_auc = None, -1.0
for r in grid_results:
    weights = {"35M": r["w35m"], "650M": r["w650m"], "3B": r["w3b"]}
    if r["val_auc"] > best_val_auc:
        best_val_auc, best_w_auc = r["val_auc"], weights
test_p_auc = sum(best_w_auc[s] * scale_test_probs[s] for s in SCALE_NAMES)
t_auc, _ = best_thresh(sum(best_w_auc[s] * scale_val_probs[s] for s in SCALE_NAMES), y_val_ref)
weighted_result_auc = metrics_at(test_p_auc, y_test_ref, t_auc)

print(f"\n{'variant':<32}{'AUC':>8}{'AUPRC':>8}{'MCC':>8}{'F1':>8}")
print("-"*64)
print(f"{'equal-weight (all-transfer)':<32}{equal_result['auc']:>8.4f}{equal_result['auprc']:>8.4f}{equal_result['mcc']:>8.4f}{equal_result['f1']:>8.4f}")
print(f"{'weighted (val-MCC-optimal)':<32}{weighted_result['auc']:>8.4f}{weighted_result['auprc']:>8.4f}{weighted_result['mcc']:>8.4f}{weighted_result['f1']:>8.4f}")
print(f"{'weighted (val-AUC-optimal)':<32}{weighted_result_auc['auc']:>8.4f}{weighted_result_auc['auprc']:>8.4f}{weighted_result_auc['mcc']:>8.4f}{weighted_result_auc['f1']:>8.4f}")
print(f"{'-- prior best (Phase131, w/ 650M-scratch) --':<32}")
print(f"{'weighted (val-MCC-optimal)':<32}{0.9145:>8.4f}{'--':>8}{0.4669:>8.4f}{'--':>8}")
print(f"{'weighted (val-AUC-optimal)':<32}{0.9157:>8.4f}{'--':>8}{0.4602:>8.4f}{'--':>8}")

json.dump({
    "solo_results": solo_results,
    "equal_weight_result": equal_result,
    "best_weights_by_val_mcc": best_w, "val_mcc": best_val_mcc,
    "best_weights_by_val_auc": best_w_auc, "val_auc": best_val_auc,
    "test_result_val_mcc_weights": weighted_result,
    "test_result_val_auc_weights": weighted_result_auc,
    "grid": grid_results,
}, open(WORK/"phase148_weighted_ensemble_alltransfer_results.json", "w"), indent=2)
print("\nsaved -> phase148_weighted_ensemble_alltransfer_results.json")
''')

md(r'''
---

## Reading this

- **Weighted (all-transfer) beats Phase131's best** -- 650M-transfer's
  quality lift (over 650M-scratch) flowed through into a genuinely
  better final ensemble. This is the new standing-best CryptoBench
  result and the natural stopping point for this improvement thread.
- **Ties or barely beats Phase131** -- the ensemble was already mostly
  saturated by 3B-transfer's strength; 650M's quality lift helps less
  than its solo numbers suggest once combined.
- **Any single scale alone beats the ensemble** -- report that scale
  alone as the final number; ensembling added no value here.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase148_cryptobench_weighted_ensemble_alltransfer.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
