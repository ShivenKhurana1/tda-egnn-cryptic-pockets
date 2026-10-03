#!/usr/bin/env python3
"""Phase 130: THE COMBINED CryptoBench result -- all 3 confirmed wins at
once. Topology: already baked into every model (nothing to toggle).
Transfer learning: 35M uses Phase122+128's transfer-learned 8 seeds
(+0.0218 val PR confirmed); 3B uses Phase129's transfer-learned 8 seeds
(new, no matching main-pipeline checkpoint exists at 650M so that scale
stays from-scratch, Phase46/47). Cross-scale ensemble: combines all 3
scales' predictions, val-tuned threshold (Phase124's discipline).
Compares against the prior best (all-from-scratch ensemble: AUC 0.9000,
MCC 0.4260)."""
import json
from pathlib import Path

nb124 = json.load(open("/Users/shivenkhurana/bioproj/phase124_cryptobench_cross_scale_ensemble_valtuned.ipynb"))
c124 = nb124["cells"]
s124 = lambda i: c124[i]["source"]

C=[]
md=lambda s:C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
code=lambda s:C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r'''
# Phase 130 - CryptoBench: All 3 Confirmed Wins Combined

Topology (baked into every model), transfer learning (35M: Phase122+128,
+0.0218 val PR confirmed; 3B: Phase129, new), cross-scale ensemble
(Phase123/124 pattern). 650M has no matching main-pipeline checkpoint so
stays from-scratch. Compares against the prior best all-from-scratch
ensemble (AUC 0.9000, MCC 0.4260, Phase124).
''')

md(r'''## Cell 1 - Install dependencies''')
code(s124(2))

md(r'''## Cell 2 - Model (identical to every CryptoBench phase)''')
code(s124(4))

md(r'''## Cell 3 - Load each scale''' + "'" + r'''s blob + 8 checkpoints (35M and 3B: transfer-learned; 650M: from-scratch), predict on VAL and TEST''')
code(r'''
SCALES = {
    "35M":  dict(blob="phase42_cryptobench_graphs.pt",       ckpt_prefix="cryptobench_transfer_seed"),
    "650M": dict(blob="phase45_cryptobench_graphs_esm650m.pt", ckpt_prefix="cryptobench_esm650m_deploy_seed"),
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

md(r'''## Cell 4 - Cross-scale ensemble, threshold tuned on VAL ONLY, applied to TEST''')
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
    t = best_thresh(val_p, y_val_ref)
    return metrics_at(test_p, y_test_ref, t), t

results = {}
for name, scales in [("35M-transfer", ["35M"]), ("650M-scratch", ["650M"]), ("3B-transfer", ["3B"]),
                      ("35M+650M", ["35M","650M"]), ("650M+3B", ["650M","3B"]),
                      ("35M+650M+3B (COMBINED)", ["35M","650M","3B"])]:
    m, t = combo(scales)
    results[name] = {**m, "val_thresh": float(t)}

print(f"{'variant':<24}{'AUC':>8}{'AUPRC':>8}{'MCC':>8}{'F1':>8}{'val_thresh':>12}")
print("-"*68)
for name, m in results.items():
    print(f"{name:<24}{m['auc']:>8.4f}{m['auprc']:>8.4f}{m['mcc']:>8.4f}{m['f1']:>8.4f}{m['val_thresh']:>12.2f}")

print(f"\nfor reference, the prior BEST (all-from-scratch, Phase124):")
print(f"  35M+650M+3B: AUC 0.9000  MCC 0.4190")
print(f"  650M+3B (best MCC combo): AUC 0.8974  MCC 0.4260")

json.dump(results, open(WORK/"phase130_combined_results.json", "w"), indent=2)
print("\nsaved -> phase130_combined_results.json")
''')

md(r'''
---

## Reading this

- **Combined beats Phase124''' + "'" + r'''s prior best (AUC>0.9000 and/or MCC>0.4260)**
  -- the 3 wins compound, a genuinely new best CryptoBench result.
- **Combined ties Phase124''' + "'" + r'''s prior best** -- transfer learning helps each
  scale individually but doesn'''+"'"+r'''t add beyond what the ensemble already
  captured (the wins may be correlated, not additive).
- **Combined is WORSE than Phase124** -- would be surprising; check
  whether 3B-transfer''' + "'" + r'''s individual val PR (Phase129) is actually
  competitive with 3B-scratch before trusting the ensemble combination.
''')

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11","mimetype":"text/x-python",
    "codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3",
    "nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":5}
o=Path("/Users/shivenkhurana/bioproj/phase130_cryptobench_combined.ipynb")
o.write_text(json.dumps(nb,indent=1)+"\n")
print(f"wrote {o} ({o.stat().st_size/1024:.1f} KB, {len(C)} cells)")
