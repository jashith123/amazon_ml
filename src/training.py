"""Stage 7-9 (Person D): two-phase LightGBM training, hard-negative mining,
isotonic calibration, and validation macro-F0.5 (incl. leave-one-country-out).

Inputs  (Contract 3): data/features/feature_matrix_train.parquet (+ label),
                      data/splits/{train,val,val_us_only,val_india_only}_entity_ids.txt
Outputs (Contract 4): models/lgbm_phase1.txt, models/lgbm_phase2.txt, models/isotonic_calibrator.pkl

CLI:
  python src/training.py                       # real data
  python src/training.py --fixture             # samples/ fixture smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import schema as S  # noqa: E402
from decision import apply_hard_vetoes, decide, macro_f05  # noqa: E402

LGB_PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.05, num_leaves=63, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=-1, verbose=-1, seed=S.RANDOM_SEED,
)


def read_ids(path: Path) -> set[str]:
    if not Path(path).exists():
        return set()
    return {ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()}


def load_truth_from_features(feats: pd.DataFrame) -> dict[str, set]:
    """Ground truth restricted to candidate pairs (blocking misses are added by caller)."""
    pos = feats.loc[feats[S.LABEL_COL] == 1, ["source1_entity_id", "candidate_entity_id"]]
    return pos.groupby("source1_entity_id")["candidate_entity_id"].agg(set).to_dict()


def load_full_truth(gt_path: Path) -> dict[str, set]:
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    return {s: ({x for x in m.split(",") if x} if m else set()) for s, m in zip(gt[S.GT_COLS[0]], gt[S.GT_COLS[1]])}


def fit_lgbm(X_tr, y_tr, w_tr, X_va, y_va, n_estimators: int, feature_names: list[str],
             early_stopping: int = 50) -> lgb.Booster:
    dtr = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=feature_names, free_raw_data=False)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr, feature_name=feature_names, free_raw_data=False)
    return lgb.train(LGB_PARAMS, dtr, num_boost_round=n_estimators, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(early_stopping, verbose=False), lgb.log_evaluation(200)])


def evaluate(feats: pd.DataFrame, probs: np.ndarray, entities: set[str], truth: dict[str, set],
             label: str, formula: str = "full") -> dict:
    mask = feats["source1_entity_id"].isin(entities).to_numpy()
    if mask.sum() == 0:
        return {}
    pairs = feats.loc[mask, ["source1_entity_id", "candidate_entity_id"]].copy()
    pairs["prob"] = probs[mask]
    res = {"n_entities": len(entities)}
    pred = decide(pairs, formula=formula, exclusivity=True)
    res[f"f05_{formula}"] = macro_f05(pred, truth, entities)
    pred_noex = decide(pairs, formula=formula, exclusivity=False)
    res[f"f05_{formula}_noexcl"] = macro_f05(pred_noex, truth, entities)
    best_thr, best = 0.5, -1.0
    for thr in np.arange(0.3, 0.96, 0.05):
        pred_t = decide(pairs, formula="pdf", exclusivity=True, min_prob=float(thr))
        f = macro_f05(pred_t, truth, entities)
        if f > best:
            best, best_thr = f, float(thr)
    res["f05_threshold_best"] = best
    res["threshold_best"] = best_thr
    print(f"  [{label}] " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()))
    return res


def run(feature_path: Path, splits_dir: Path, models_dir: Path, gt_path: Path | None,
        n_estimators: int = 2000, hard_neg_thr: float = 0.30, hard_neg_weight: float = 2.0,
        neg_ratio_random: float = 1.0, formula: str = "full") -> dict:
    t0 = time.time()
    feats = pd.read_parquet(feature_path)
    if S.LABEL_COL not in feats.columns:
        raise ValueError("training feature matrix must contain the 'label' column")
    feat_cols = S.feature_columns(feats.columns)
    S.assert_no_raw_counts(feat_cols)
    if feats[feat_cols].isna().any().any():
        raise ValueError("NaNs in feature matrix (Contract 3 forbids NaN)")
    print(f"features: {len(feats):,} pairs, {len(feat_cols)} feature cols, "
          f"pos rate {feats[S.LABEL_COL].mean():.3f}, loaded in {time.time()-t0:.0f}s")

    train_ids = read_ids(splits_dir / S.SPLIT_TRAIN_IDS.name)
    val_ids = read_ids(splits_dir / S.SPLIT_VAL_IDS.name)
    val_us = read_ids(splits_dir / S.SPLIT_VAL_US_IDS.name)
    val_in = read_ids(splits_dir / S.SPLIT_VAL_INDIA_IDS.name)
    all_ents = feats["source1_entity_id"].unique()
    if not train_ids or not val_ids:
        rng = np.random.RandomState(S.RANDOM_SEED)
        ents = rng.permutation(all_ents)
        n_val = max(1, int(0.2 * len(ents)))
        val_ids, train_ids = set(ents[:n_val]), set(ents[n_val:])
        print("split files missing: using an 80/20 entity split generated here")
    # calibration fold carved out of the training entities (entity level)
    rng = np.random.RandomState(S.RANDOM_SEED)
    tr_list = np.array(sorted(train_ids))
    cal_ids = set(rng.choice(tr_list, size=max(1, int(0.15 * len(tr_list))), replace=False))
    fit_ids = train_ids - cal_ids
    is_fit = feats["source1_entity_id"].isin(fit_ids).to_numpy()
    is_cal = feats["source1_entity_id"].isin(cal_ids).to_numpy()
    is_val = feats["source1_entity_id"].isin(val_ids).to_numpy()
    X = feats[feat_cols].to_numpy(np.float32)
    y = feats[S.LABEL_COL].to_numpy(np.int8)
    print(f"entities: fit={len(fit_ids):,} cal={len(cal_ids):,} val={len(val_ids):,}")

    # ---- Phase 1: positives + random negatives -------------------------------
    rng = np.random.RandomState(S.RANDOM_SEED)
    pos_idx = np.where(is_fit & (y == 1))[0]
    neg_idx = np.where(is_fit & (y == 0))[0]
    n_rand = min(len(neg_idx), int(neg_ratio_random * len(pos_idx)))
    rand_neg = rng.choice(neg_idx, size=n_rand, replace=False)
    p1_idx = np.concatenate([pos_idx, rand_neg])
    print(f"phase 1: {len(pos_idx):,} pos + {len(rand_neg):,} random neg")
    m1 = fit_lgbm(X[p1_idx], y[p1_idx], np.ones(len(p1_idx), np.float32), X[is_val], y[is_val], n_estimators, feat_cols)
    models_dir.mkdir(parents=True, exist_ok=True)
    m1.save_model(str(models_dir / S.MODEL_PHASE1.name))
    p1_all = m1.predict(X)
    print(f"phase 1 val AUC={roc_auc_score(y[is_val], p1_all[is_val]):.4f} (best_iter={m1.best_iteration})")

    # ---- Phase 2: hard-negative mining (2 : 1 : 1 = pos : hard : random) -----
    hard_idx = neg_idx[p1_all[neg_idx] >= hard_neg_thr]
    n_hard_target = max(1, len(pos_idx) // 2)
    if len(hard_idx) > n_hard_target:
        hard_idx = hard_idx[np.argsort(-p1_all[hard_idx])[:n_hard_target]]
    rand2 = rng.choice(neg_idx, size=min(len(neg_idx), n_hard_target), replace=False)
    p2_idx = np.concatenate([pos_idx, hard_idx, rand2])
    w2 = np.concatenate([np.ones(len(pos_idx)), np.full(len(hard_idx), hard_neg_weight), np.ones(len(rand2))]).astype(np.float32)
    print(f"phase 2: {len(pos_idx):,} pos + {len(hard_idx):,} hard neg (x{hard_neg_weight}) + {len(rand2):,} random neg")
    m2 = fit_lgbm(X[p2_idx], y[p2_idx], w2, X[is_val], y[is_val], n_estimators, feat_cols)
    m2.save_model(str(models_dir / S.MODEL_PHASE2.name))
    p2_all = m2.predict(X)
    print(f"phase 2 val AUC={roc_auc_score(y[is_val], p2_all[is_val]):.4f} (best_iter={m2.best_iteration})")

    # ---- Isotonic calibration on the held-out calibration entities ------------
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    cal.fit(p2_all[is_cal], y[is_cal])
    joblib.dump(cal, models_dir / S.CALIBRATOR.name)
    p_cal = cal.predict(p2_all)
    bins = np.linspace(0, 1, 11)
    dig = np.digitize(p_cal[is_val], bins) - 1
    print("reliability (val): bin -> mean_pred / frac_pos / n")
    for b in range(10):
        m = dig == b
        if m.sum():
            print(f"  {bins[b]:.1f}-{bins[b+1]:.1f}: {p_cal[is_val][m].mean():.3f} / {y[is_val][m].mean():.3f} / {m.sum()}")
    p_final = apply_hard_vetoes(feats, p_cal)

    # ---- Validation macro-F0.5 (blocking misses count as FN) ------------------
    truth = load_truth_from_features(feats)
    if gt_path and Path(gt_path).exists():
        full = load_full_truth(gt_path)
        truth = {e: full.get(e, set()) for e in all_ents}
    results = {}
    for name, ents in (("val", val_ids), ("val_us_only", val_us), ("val_india_only", val_in)):
        if ents:
            results[name] = evaluate(feats, p_final, ents & set(all_ents), truth, name, formula)
    imp = pd.Series(m2.feature_importance("gain"), index=feat_cols).sort_values(ascending=False)
    print("top features (gain):")
    print(imp.head(15).round(0).to_string())
    summary = {"feature_cols": feat_cols, "results": results, "phase1_best_iter": m1.best_iteration,
               "phase2_best_iter": m2.best_iteration, "formula": formula}
    (models_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"done in {time.time()-t0:.0f}s; models in {models_dir}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 7-9: two-phase LightGBM + calibration")
    ap.add_argument("--features", default=str(S.FEATURES_TRAIN))
    ap.add_argument("--splits-dir", default=str(S.SPLITS_DIR))
    ap.add_argument("--models-dir", default=str(S.MODELS_DIR))
    ap.add_argument("--ground-truth", default=str(S.dataset_dir() / "train" / "train_ground_truth.tsv"))
    ap.add_argument("--n-estimators", type=int, default=2000)
    ap.add_argument("--hard-neg-thr", type=float, default=0.30)
    ap.add_argument("--hard-neg-weight", type=float, default=2.0)
    ap.add_argument("--formula", default="full", choices=["full", "pdf"])
    ap.add_argument("--fixture", action="store_true", help="run on samples/ fixtures")
    args = ap.parse_args(argv)
    if args.fixture:
        run(S.SAMPLES_DIR / "feature_matrix_train.parquet", S.SAMPLES_DIR / "splits",
            S.SAMPLES_DIR / "models", None, n_estimators=200, formula=args.formula)
    else:
        run(Path(args.features), Path(args.splits_dir), Path(args.models_dir), Path(args.ground_truth),
            n_estimators=args.n_estimators, hard_neg_thr=args.hard_neg_thr,
            hard_neg_weight=args.hard_neg_weight, formula=args.formula)


if __name__ == "__main__":
    main()
