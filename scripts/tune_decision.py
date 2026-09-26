"""Tune the decision layer on the validation entities of the trained model.

Loads the train feature matrix, restricts to val entities, scores with the phase-2 model +
isotonic calibrator, then evaluates macro-F0.5 for combinations of: selection formula
(full / pdf), probability floor, hard vetoes, exclusivity, and per-country calibration.

  python scripts/tune_decision.py [--features data/features/feature_matrix_train.parquet]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import schema as S  # noqa: E402
from decision import apply_hard_vetoes, decide, macro_f05  # noqa: E402
from training import load_full_truth, read_ids  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(S.FEATURES_TRAIN))
    ap.add_argument("--models-dir", default=str(S.MODELS_DIR))
    args = ap.parse_args()
    t0 = time.time()
    md = Path(args.models_dir)
    booster = lgb.Booster(model_file=str(md / S.MODEL_PHASE2.name))
    cols = booster.feature_name()
    val_ids = read_ids(S.SPLITS_DIR / S.SPLIT_VAL_IDS.name)
    cal_ids_all = read_ids(S.SPLITS_DIR / S.SPLIT_TRAIN_IDS.name)
    feats = pd.read_parquet(args.features)
    is_val = feats["source1_entity_id"].isin(val_ids).to_numpy()
    # calibration entities: same 15% carve-out as training.py
    rng = np.random.RandomState(S.RANDOM_SEED)
    tr_list = np.array(sorted(cal_ids_all))
    cal_ids = set(rng.choice(tr_list, size=max(1, int(0.15 * len(tr_list))), replace=False))
    is_cal = feats["source1_entity_id"].isin(cal_ids).to_numpy()
    sub = feats[is_val | is_cal].reset_index(drop=True)
    is_val = sub["source1_entity_id"].isin(val_ids).to_numpy()
    is_cal = ~is_val
    del feats
    raw = booster.predict(sub[cols].to_numpy(np.float32))
    y = sub[S.LABEL_COL].to_numpy(np.int8)
    print(f"scored {len(sub):,} pairs (val {is_val.sum():,}, cal {is_cal.sum():,}) in {time.time()-t0:.0f}s", flush=True)

    truth_full = load_full_truth(S.dataset_dir() / "train" / "train_ground_truth.tsv")
    ents = sorted(val_ids)
    truth = {e: truth_full.get(e, set()) for e in ents}
    country = sub.drop_duplicates("source1_entity_id").set_index("source1_entity_id")["country"]
    ents_by_c = {c: [e for e in ents if country.get(e) == c] for c in country.unique()}

    cal_global = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw[is_cal], y[is_cal])
    p_global = cal_global.predict(raw)
    p_country = p_global.copy()
    for c in country.unique():
        m_c = (sub["country"] == c).to_numpy()
        if (m_c & is_cal).sum() > 1000:
            cal_c = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw[m_c & is_cal], y[m_c & is_cal])
            p_country[m_c] = cal_c.predict(raw[m_c])

    def run(p, formula, min_prob, vetoes, excl, label):
        pv = apply_hard_vetoes(sub, p) if vetoes else p
        pairs = sub.loc[is_val, ["source1_entity_id", "candidate_entity_id"]].copy()
        pairs["prob"] = pv[is_val]
        pred = decide(pairs, formula=formula, exclusivity=excl, min_prob=min_prob)
        res = {"all": macro_f05(pred, truth, ents)}
        for c, es in ents_by_c.items():
            res[c] = macro_f05(pred, truth, es)
        print(f"{label:48s} " + "  ".join(f"{k}={v:.4f}" for k, v in res.items()), flush=True)
        return res["all"]

    results = []
    for cal_name, p in (("global-cal", p_global), ("country-cal", p_country)):
        for formula in ("full", "pdf"):
            for min_prob in (0.0, 0.02, 0.05, 0.1):
                for vetoes in (True, False):
                    if not vetoes and (min_prob != 0.0 or formula != "full"):
                        continue
                    f = run(p, formula, min_prob, vetoes, True, f"{cal_name} {formula} floor={min_prob} vetoes={int(vetoes)}")
                    results.append((f, cal_name, formula, min_prob, vetoes))
    run(p_global, "full", 0.0, True, False, "global-cal full floor=0 vetoes=1 NO-exclusivity")
    run(raw, "full", 0.0, True, True, "UNcalibrated full floor=0 vetoes=1")
    best = max(results)
    print("BEST:", best, f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
