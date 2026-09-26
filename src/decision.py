"""Stage 10-11 (Person D): F0.5-optimised decision layer + output writing.

Implements, per the pipeline document (section 5 / section 10):
  * expected-F0.5 subset selection per Source 1 entity (closed form)
  * exclusivity constraint (each S2/S3 id claimed by at most one S1)
  * hard vetoes (house-number conflict + low address similarity; low name + low embedding sim)
  * macro-F0.5 scorer (singletons included)
  * writers for output/matching_results.tsv and output/candidate_pairs.tsv

CLI (test-time):
  python src/decision.py --features data/features/feature_matrix_test.parquet \
      --candidates data/candidates/candidate_pairs_test.parquet \
      --source1 dataset/test/test_source1.tsv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import schema as S  # noqa: E402

# --------------------------------------------------------------------------- #
# Expected-F0.5 selection
# --------------------------------------------------------------------------- #

def expected_f05_select(probs, formula: str = "full") -> int:
    """Return k (number of top candidates to accept) maximising expected F0.5.

    formula='pdf'  : the document's closed form 1.25*S_k / (k + 0.25*S_k), k=0 scores 0.
    formula='full' : same numerator, but the denominator uses the expected number of TRUE
                     matches among all candidates (S_all), and k=0 scores prod(1-p), i.e.
                     the probability the entity really is a singleton.  This is the exact
                     expectation-ratio for the per-entity F0.5 = 1.25*TP / (k + 0.25*T).
    """
    probs = np.sort(np.asarray(probs, dtype=np.float64))[::-1]
    if len(probs) == 0:
        return 0
    s_all = probs.sum()
    if formula == "pdf":
        best_k, best = 0, 0.0
    else:
        best_k, best = 0, float(np.prod(1.0 - probs))
    cum = 0.0
    for k, p in enumerate(probs, 1):
        cum += p
        denom = k + 0.25 * (cum if formula == "pdf" else s_all)
        ef = 1.25 * cum / denom
        if ef > best:
            best, best_k = ef, k
    return best_k


def select_matches(pairs: pd.DataFrame, prob_col: str = "prob", formula: str = "full",
                   min_prob: float = 0.0) -> pd.DataFrame:
    """Vectorised per-entity expected-F0.5 selection. Returns the accepted rows."""
    df = pairs[["source1_entity_id", "candidate_entity_id", prob_col]].copy()
    df = df[df[prob_col] >= min_prob]
    if df.empty:
        return df
    df.sort_values(["source1_entity_id", prob_col], ascending=[True, False], inplace=True, ignore_index=True)
    p = df[prob_col].to_numpy(np.float64)
    grp = df.groupby("source1_entity_id", sort=False)
    k = grp.cumcount().to_numpy() + 1
    cum = grp[prob_col].cumsum().to_numpy(np.float64)
    if formula == "pdf":
        ef = 1.25 * cum / (k + 0.25 * cum)
        ef0 = pd.Series(0.0, index=df["source1_entity_id"].unique())
    else:
        s_all = grp[prob_col].transform("sum").to_numpy(np.float64)
        ef = 1.25 * cum / (k + 0.25 * s_all)
        ef0 = grp[prob_col].apply(lambda s: float(np.prod(1.0 - s.to_numpy(np.float64))))
    df["_ef"] = ef
    df["_k"] = k
    best = df.loc[grp["_ef"].idxmax(), ["source1_entity_id", "_ef", "_k"]].set_index("source1_entity_id")
    best_k = best["_k"].where(best["_ef"] > ef0.reindex(best.index).to_numpy(), 0)
    keep_k = df["source1_entity_id"].map(best_k).to_numpy()
    out = df[df["_k"] <= keep_k].drop(columns=["_ef", "_k"])
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Exclusivity + vetoes
# --------------------------------------------------------------------------- #

def enforce_exclusivity(selected: pd.DataFrame, prob_col: str = "prob") -> pd.DataFrame:
    """Each S2/S3 id is kept only for the S1 that claims it with the highest probability."""
    if selected.empty:
        return selected
    out = selected.sort_values(prob_col, ascending=False, kind="stable")
    out = out.drop_duplicates("candidate_entity_id", keep="first")
    return out.sort_values(["source1_entity_id", prob_col], ascending=[True, False], ignore_index=True)


def enforce_exclusivity_dict(matches_dict: dict[str, list[tuple[str, float]]]) -> dict[str, list[str]]:
    """Document section 10 version (dict in, dict out)."""
    claimed: dict[str, str] = {}
    result: dict[str, list[str]] = {s1: [] for s1 in matches_dict}
    all_pairs = [(p, s1, s23) for s1, pairs in matches_dict.items() for s23, p in pairs]
    for prob, s1, s23 in sorted(all_pairs, reverse=True):
        if s23 not in claimed:
            claimed[s23] = s1
            result[s1].append(s23)
    return result


def apply_hard_vetoes(features: pd.DataFrame, probs: np.ndarray,
                      addr_sim_max: float = 0.30, name_sim_max: float = 0.30,
                      emb_sim_max: float = 0.40) -> np.ndarray:
    """Zero out the probability of pairs failing the document's hard vetoes.

    Veto 1: house-number conflict AND address similarity < addr_sim_max.
    Veto 2: name similarity < name_sim_max AND embedding cosine < emb_sim_max
            (skipped when the embedding column is absent or all-zero, per team plan).
    """
    p = probs.copy()
    cols = features.columns
    if S.FEAT_HOUSE_CONFLICT in cols and S.FEAT_ADDR_SIM in cols:
        v1 = (features[S.FEAT_HOUSE_CONFLICT].to_numpy() > 0) & (features[S.FEAT_ADDR_SIM].to_numpy() < addr_sim_max)
        p[v1] = 0.0
    if S.FEAT_NAME_SIM in cols and S.FEAT_EMB_COS in cols:
        emb = features[S.FEAT_EMB_COS].to_numpy()
        if np.any(emb != 0):
            v2 = (features[S.FEAT_NAME_SIM].to_numpy() < name_sim_max) & (emb < emb_sim_max)
            p[v2] = 0.0
    return p


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def f05_entity(pred: set, truth: set) -> float:
    if not truth:
        return 1.0 if not pred else 0.0
    tp = len(pred & truth)
    fp = len(pred) - tp
    fn = len(truth) - tp
    if tp == 0:
        return 0.0
    return 1.25 * tp / (1.25 * tp + 0.25 * fn + fp)


def macro_f05(pred: dict[str, set], truth: dict[str, set], entities) -> float:
    """Macro F0.5 over `entities`; entities missing from `pred` count as empty predictions."""
    entities = list(entities)
    if not entities:
        return float("nan")
    return float(np.mean([f05_entity(pred.get(e, set()), truth.get(e, set())) for e in entities]))


def selected_to_dict(selected: pd.DataFrame) -> dict[str, set]:
    if selected.empty:
        return {}
    return selected.groupby("source1_entity_id")["candidate_entity_id"].agg(set).to_dict()


def decide(pairs: pd.DataFrame, prob_col: str = "prob", formula: str = "full",
           exclusivity: bool = True, min_prob: float = 0.0) -> dict[str, set]:
    """Full decision layer: selection -> exclusivity -> dict s1 -> set(ids)."""
    sel = select_matches(pairs, prob_col=prob_col, formula=formula, min_prob=min_prob)
    if exclusivity:
        sel = enforce_exclusivity(sel, prob_col=prob_col)
    return selected_to_dict(sel)


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #

def _join_ids(ids) -> str:
    seen, out = set(), []
    for x in ids:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return ",".join(out)


def write_matching_results(pred: dict[str, set], all_s1_ids, path: Path = S.MATCHING_RESULTS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [(s1, _join_ids(sorted(pred.get(s1, ())))) for s1 in all_s1_ids]
    pd.DataFrame(rows, columns=S.MATCHING_HEADER).to_csv(path, sep="\t", index=False)


def write_candidate_pairs(candidates: pd.DataFrame, all_s1_ids, path: Path = S.CANDIDATE_PAIRS_TSV) -> None:
    """Collapse Contract 2 (one row per pair) to one row per S1 entity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped = candidates.groupby("source1_entity_id")["candidate_entity_id"].agg(_join_ids)
    rows = [(s1, grouped.get(s1, "")) for s1 in all_s1_ids]
    pd.DataFrame(rows, columns=S.CANDIDATE_HEADER).to_csv(path, sep="\t", index=False)


# --------------------------------------------------------------------------- #
# Inference on the test feature matrix
# --------------------------------------------------------------------------- #

def predict_probs(features: pd.DataFrame, model_path: Path, calibrator_path: Path | None,
                  vetoes: bool = True) -> np.ndarray:
    booster = lgb.Booster(model_file=str(model_path))
    cols = booster.feature_name()
    missing = [c for c in cols if c not in features.columns]
    if missing:
        raise ValueError(f"feature matrix is missing model columns: {missing[:10]}")
    raw = booster.predict(features[cols].to_numpy(np.float32))
    if calibrator_path and Path(calibrator_path).exists():
        raw = joblib.load(calibrator_path).predict(raw)
    if vetoes:
        raw = apply_hard_vetoes(features, raw)
    return np.clip(raw, 0.0, 1.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 10-11: decision layer + outputs")
    ap.add_argument("--features", default=str(S.FEATURES_TEST))
    ap.add_argument("--candidates", default=str(S.CANDIDATES_TEST))
    ap.add_argument("--source1", default=str(S.dataset_dir() / "test" / "test_source1.tsv"))
    ap.add_argument("--model", default=str(S.MODEL_PHASE2))
    ap.add_argument("--calibrator", default=str(S.CALIBRATOR))
    ap.add_argument("--formula", default="full", choices=["full", "pdf"])
    ap.add_argument("--min-prob", type=float, default=0.0)
    ap.add_argument("--no-vetoes", action="store_true")
    ap.add_argument("--no-exclusivity", action="store_true")
    ap.add_argument("--out-dir", default=str(S.OUTPUT_DIR))
    args = ap.parse_args(argv)

    feats = pd.read_parquet(args.features)
    cands = pd.read_parquet(args.candidates, columns=["source1_entity_id", "candidate_entity_id"])
    s1_ids = pd.read_csv(args.source1, sep="\t", dtype=str, keep_default_na=False, usecols=["entity_id"])["entity_id"].tolist()

    model_path = Path(args.model)
    if not model_path.exists() and S.MODEL_PHASE1.exists():
        print(f"{model_path} not found, falling back to {S.MODEL_PHASE1}")
        model_path = S.MODEL_PHASE1
    probs = predict_probs(feats, model_path, Path(args.calibrator), vetoes=not args.no_vetoes)
    pairs = feats[["source1_entity_id", "candidate_entity_id"]].copy()
    pairs["prob"] = probs
    pred = decide(pairs, formula=args.formula, exclusivity=not args.no_exclusivity, min_prob=args.min_prob)

    out_dir = Path(args.out_dir)
    write_matching_results(pred, s1_ids, out_dir / "matching_results.tsv")
    write_candidate_pairs(cands, s1_ids, out_dir / "candidate_pairs.tsv")
    n_match = sum(len(v) for v in pred.values())
    n_single = sum(1 for s in s1_ids if not pred.get(s))
    print(f"wrote {out_dir/'matching_results.tsv'}: {len(s1_ids):,} entities, {n_match:,} matches, "
          f"{n_single:,} singletons ({n_single/len(s1_ids):.1%})")
    print(f"wrote {out_dir/'candidate_pairs.tsv'}: {len(cands):,} candidate pairs")


if __name__ == "__main__":
    main()
