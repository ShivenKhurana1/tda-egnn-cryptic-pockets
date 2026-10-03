#!/usr/bin/env python3
"""Phase 124: redo of Phase123's cross-scale ensemble with a PROPER
val-tuned threshold (Phase123 tuned on test itself for a quick same-run
comparison -- AUC is threshold-independent so that number stands, but
MCC/F1 needed the val-tuned redo before citing). Same 3 scales x 8 seeds,
same alignment-verification discipline, now predicting on val AND test:
tune per-combination threshold on VAL predictions only, then apply that
fixed threshold to TEST -- matching exactly how Phase44/47/55 tuned their
own single-scale numbers (val-tuned, test-evaluated, never test-tuned)."""
import json
from pathlib import Path

nb123 = json.load(open("/Users/shivenkhurana/bioproj/phase123_cryptobench_cross_scale_ensemble.ipynb"))
c123 = nb123["cells"]
s123 = lambda i: c123[i]["source"]

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 124 - Cross-Scale Ensemble, PROPER Val-Tuned Threshold

Phase123 found AUC 0.9000 for the 35M+650M+3B ensemble (threshold-
independent, stands as-is) but tuned its MCC/F1 threshold on TEST itself
for a quick same-run check -- methodologically loose. This redoes it
correctly: predicts on VAL too, tunes the decision threshold there only,
then applies that fixed threshold to TEST -- identical discipline to
every single-scale number in this project (Phase44/47/55).
''')

md(r'''## Cell 1 - Install dependencies''')
code(s123(2))

md(r'''## Cell 2 - Model (identical to Phase123)''')
code(s123(4))

md(r'''## Cell 3 - Load each scale's blob + all 8 checkpoints, predict on VAL and TEST (pure inference, no training)''')
code(r'''
SCALES = {
    "35M":  dict(blob="phase42_cryptobench_graphs.pt",       ckpt_prefix="cryptobench_deploy_seed"),
    "650M": dict(blob="phase45_cryptobench_graphs_esm650m.pt", ckpt_prefix="cryptobench_esm650m_deploy_seed"),
    "3B":   dict(blob="phase48_cryptobench_graphs_esm3b.pt",   ckpt_prefix="cryptobench_esm3b_deploy_seed"),
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
    print(f"  {scale:<6} 8-seed val+test predictions ready "
          f"(val {len(val_set)} chains/{len(scale_val_y[scale])} res, "
          f"test {len(test_set)} chains/{len(scale_test_y[scale])} res)  [{(time.time()-t0)/60:.1f}m]")

for split_name, keys_dict in [("val", scale_val_keys), ("test", scale_test_keys)]:
    ref = keys_dict["35M"]
    for scale in ("650M", "3B"):
        assert keys_dict[scale] == ref, f"{split_name}: {scale} key order mismatch vs 35M"
print(f"\nalignment verified on both val and test across all 3 scales")
''')

md(r'''## Cell 4 - Cross-scale ensemble, threshold tuned on VAL ONLY, applied to TEST (matches Phase44/47/55''' + "'" + r'''s own discipline)''')
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
    return best_t

y_val_ref = scale_val_y["35M"]
y_test_ref = scale_test_y["35M"]

def combo(scales):
    val_p = np.mean([scale_val_probs[s] for s in scales], axis=0)
    test_p = np.mean([scale_test_probs[s] for s in scales], axis=0)
    t = best_thresh(val_p, y_val_ref)   # tuned on VAL ONLY
    return metrics_at(test_p, y_test_ref, t), t

results = {}
for name, scales in [("35M", ["35M"]), ("650M", ["650M"]), ("3B", ["3B"]),
                      ("35M+650M", ["35M","650M"]), ("650M+3B", ["650M","3B"]),
                      ("35M+650M+3B", ["35M","650M","3B"])]:
    m, t = combo(scales)
    results[name] = {**m, "val_thresh": float(t)}

print(f"{'variant':<16}{'AUC':>8}{'AUPRC':>8}{'MCC':>8}{'F1':>8}{'val_thresh':>12}")
print("-"*60)
for name, m in results.items():
    print(f"{name:<16}{m['auc']:>8.4f}{m['auprc']:>8.4f}{m['mcc']:>8.4f}{m['f1']:>8.4f}{m['val_thresh']:>12.2f}")

print(f"\nfor reference, each scale's OWN prior val-tuned 8-seed number (Phase44/47/55):")
print(f"  35M:  AUC 0.8717  MCC 0.3674")
print(f"  650M: AUC 0.8873  MCC 0.3986")
print(f"  3B:   AUC 0.8878  MCC 0.4157")
print(f"\nPhase123's test-tuned (methodologically loose) numbers, for comparison:")
print(f"  35M+650M+3B: AUC 0.9000  MCC 0.4287  (test-tuned threshold)")

json.dump(results, open(WORK/"phase124_cross_scale_ensemble_valtuned.json", "w"), indent=2)
print("\nsaved -> phase124_cross_scale_ensemble_valtuned.json")
''')

md(r'''
---

## Reading this

- **35M+650M+3B still beats every individual scale on val-tuned MCC/F1,
  similar magnitude to Phase123's test-tuned numbers** -- confirms the
  ensemble gain is real, not a test-tuning artifact.
- **val-tuned MCC/F1 drops notably vs Phase123's test-tuned numbers, closer
  to (or below) the best individual scale** -- Phase123's test-tuned
  threshold was overfitting to the test set itself; the AUC gain (which
  never depended on any threshold) is the only trustworthy part of that
  result.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase124_cryptobench_cross_scale_ensemble_valtuned.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
