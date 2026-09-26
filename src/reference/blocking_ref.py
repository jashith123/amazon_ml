"""Candidate generation (blocking).

Every partition is one country (open set of labels: whatever string appears in
the data).  For a partition the gallery is the union of Source 2 and Source 3
records of that country; queries are the Source 1 records of that country.

Legs (each contributes a bit in the `legs` column of the pair table):
    1   exact key: name without spaces (catches domain-style names)
    2   exact key: sorted core-name tokens (word reordering)
    4   exact key: first two name tokens + house number
    8   exact key: full normalised address
    16  exact key: house number + street token
    32  word TF-IDF cosine on core name + address, top-N (sparse_dot_topn)
    64  character 3-4-gram TF-IDF cosine on space-less name, top-N
    128 bidirectional word TF-IDF: gallery record -> its best S1 queries

The union is scored with a cheap heuristic and cut to `final_k` per S1 record:
that cut set is exactly what the matcher scores, and what goes to
candidate_pairs.tsv.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

LEG_NOSPACE, LEG_SORTED, LEG_NAMEHOUSE, LEG_ADDR, LEG_HOUSESTREET = 1, 2, 4, 8, 16
LEG_WORD, LEG_CHAR, LEG_BIDIR = 32, 64, 128
KEY_LEGS = (LEG_NOSPACE, LEG_SORTED, LEG_NAMEHOUSE, LEG_ADDR, LEG_HOUSESTREET)


@dataclass
class BlockConfig:
    word_topn: int = 30
    word_thr: float = 0.15
    word_max_df: float = 0.01
    char_topn: int = 20
    char_thr: float = 0.30
    char_max_df: float = 0.005
    bidir: bool = True
    bidir_topn: int = 2
    bidir_thr: float = 0.30
    key_cap: int = 60
    final_k: int = 30
    threads: int = 16


def word_doc(df: pd.DataFrame) -> pd.Series:
    return (df["name_core"].astype(str) + " " + df["addr_norm"].astype(str)).str.strip()


def _key_series(df: pd.DataFrame) -> dict[int, pd.Series]:
    """Blocking keys per record; empty string means 'no key'."""
    core = df["name_core"].astype(str)
    nospace = df["name_nospace"].astype(str)
    sorted_ = df["name_sorted"].astype(str)
    addr = df["addr_norm"].astype(str)
    house = df["house_no"].astype(str)
    street = df["street"].astype(str)
    first2 = core.str.split(" ").str[:2].str.join(" ")
    keys = {
        LEG_NOSPACE: nospace.where(nospace.str.len() >= 5, ""),
        LEG_SORTED: sorted_.where(sorted_.str.contains(" "), ""),
        LEG_NAMEHOUSE: (first2 + "|" + house).where((house != "") & (first2 != ""), ""),
        LEG_ADDR: addr.where(addr.str.len() >= 12, ""),
        LEG_HOUSESTREET: (house + "|" + street).where((house != "") & (street != ""), ""),
    }
    return keys


def rowwise_dot(A: sp.csr_matrix, B: sp.csr_matrix, ia: np.ndarray, ib: np.ndarray,
                chunk: int = 1_000_000) -> np.ndarray:
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        a = A[ia[s:s + chunk]]
        b = B[ib[s:s + chunk]]
        out[s:s + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


class PartitionIndex:
    """All gallery-side structures for one country partition."""

    def __init__(self, gal: pd.DataFrame, cfg: BlockConfig, verbose: bool = True):
        self.cfg = cfg
        self.n = len(gal)
        self.word_vec = TfidfVectorizer(
            token_pattern=r"\S+", lowercase=False, sublinear_tf=True,
            min_df=1, max_df=cfg.word_max_df, dtype=np.float32,
        )
        self.G_word = self.word_vec.fit_transform(word_doc(gal)).tocsr()
        self.G_word_T = self.G_word.T.tocsr()
        self.char_vec = TfidfVectorizer(
            analyzer="char", ngram_range=(3, 4), lowercase=False, sublinear_tf=True,
            min_df=2, max_df=cfg.char_max_df, dtype=np.float32,
        )
        self.G_char = self.char_vec.fit_transform(gal["name_nospace"].astype(str)).tocsr()
        self.G_char_T = self.G_char.T.tocsr()
        # exact-key tables: key -> gallery row, buckets above key_cap dropped
        self.key_tables: dict[int, pd.DataFrame] = {}
        for leg, ks in _key_series(gal).items():
            t = pd.DataFrame({"key": ks.to_numpy(), "g": np.arange(self.n, dtype=np.int32)})
            t = t[t["key"] != ""]
            sizes = t.groupby("key")["g"].transform("size")
            t = t[sizes <= cfg.key_cap]
            self.key_tables[leg] = t.reset_index(drop=True)
        self.bidir_pairs: pd.DataFrame | None = None
        if verbose:
            print(f"    index built: gallery={self.n:,} word_vocab={len(self.word_vec.vocabulary_):,} "
                  f"char_vocab={len(self.char_vec.vocabulary_):,}", flush=True)

    def transform_queries(self, q: pd.DataFrame) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        return (self.word_vec.transform(word_doc(q)).tocsr(),
                self.char_vec.transform(q["name_nospace"].astype(str)).tocsr())

    def build_bidir(self, Q_word_all: sp.csr_matrix) -> None:
        """Gallery -> S1 retrieval over the whole partition (S1 rows indexed 0..len-1)."""
        cfg = self.cfg
        C = sp_matmul_topn(self.G_word, Q_word_all.T.tocsr(), top_n=cfg.bidir_topn,
                           threshold=cfg.bidir_thr, sort=True, n_threads=cfg.threads).tocoo()
        self.bidir_pairs = pd.DataFrame({"s1": C.col.astype(np.int32), "g": C.row.astype(np.int32)})
        self.bidir_pairs.sort_values("s1", inplace=True, ignore_index=True)

    def block_chunk(self, q: pd.DataFrame, s1_offset: int,
                    Q_word: sp.csr_matrix | None = None, Q_char: sp.csr_matrix | None = None) -> pd.DataFrame:
        """Candidates for a chunk of S1 rows (partition-local indices s1_offset..)."""
        cfg = self.cfg
        nq = len(q)
        if Q_word is None or Q_char is None:
            Q_word, Q_char = self.transform_queries(q)
        parts: list[tuple[np.ndarray, np.ndarray, int]] = []

        C = sp_matmul_topn(Q_word, self.G_word_T, top_n=cfg.word_topn, threshold=cfg.word_thr,
                           sort=True, n_threads=cfg.threads).tocoo()
        parts.append((C.row.astype(np.int64), C.col.astype(np.int64), LEG_WORD))
        D = sp_matmul_topn(Q_char, self.G_char_T, top_n=cfg.char_topn, threshold=cfg.char_thr,
                           sort=True, n_threads=cfg.threads).tocoo()
        parts.append((D.row.astype(np.int64), D.col.astype(np.int64), LEG_CHAR))

        for leg, ks in _key_series(q).items():
            qt = pd.DataFrame({"key": ks.to_numpy(), "s1": np.arange(nq, dtype=np.int32)})
            qt = qt[qt["key"] != ""]
            if len(qt) == 0:
                continue
            m = qt.merge(self.key_tables[leg], on="key", how="inner")
            parts.append((m["s1"].to_numpy(np.int64), m["g"].to_numpy(np.int64), leg))

        if self.bidir_pairs is not None:
            lo, hi = s1_offset, s1_offset + nq
            b = self.bidir_pairs
            sl = b.iloc[np.searchsorted(b["s1"].to_numpy(), lo):np.searchsorted(b["s1"].to_numpy(), hi)]
            if len(sl):
                parts.append((sl["s1"].to_numpy(np.int64) - lo, sl["g"].to_numpy(np.int64), LEG_BIDIR))

        s1 = np.concatenate([p[0] for p in parts])
        g = np.concatenate([p[1] for p in parts])
        legs = np.concatenate([np.full(len(p[0]), p[2], dtype=np.int32) for p in parts])
        key = s1 * self.n + g
        order = np.argsort(key, kind="stable")
        key, legs = key[order], legs[order]
        uniq, start = np.unique(key, return_index=True)
        legs_u = np.bitwise_or.reduceat(legs, start)
        s1_u = (uniq // self.n).astype(np.int32)
        g_u = (uniq % self.n).astype(np.int32)

        word_cos = rowwise_dot(Q_word, self.G_word, s1_u, g_u)
        char_cos = rowwise_dot(Q_char, self.G_char, s1_u, g_u)
        n_keys = np.zeros(len(legs_u), dtype=np.float32)
        for leg in KEY_LEGS:
            n_keys += (legs_u & leg) > 0
        heur = word_cos + char_cos + 0.25 * np.minimum(n_keys, 3) + 0.15 * ((legs_u & LEG_BIDIR) > 0)

        df = pd.DataFrame({
            "s1": s1_u + s1_offset, "g": g_u, "legs": legs_u,
            "word_cos": word_cos, "char_cos": char_cos, "heur": heur.astype(np.float32),
        })
        df.sort_values(["s1", "heur"], ascending=[True, False], inplace=True, ignore_index=True)
        df["rank"] = df.groupby("s1").cumcount().astype(np.int16)
        return df[df["rank"] < cfg.final_k].reset_index(drop=True)


def blocking_recall_report(pairs: pd.DataFrame, s1_ids: np.ndarray, g_ids: np.ndarray,
                           gt: dict[str, set[str]], ks=(5, 10, 15, 20, 25, 30, 40)) -> dict:
    """Recall of true matches inside the candidate set, at several per-S1 cut-offs."""
    truth_pairs = 0
    s1_set = set(s1_ids.tolist())
    for s, ms in gt.items():
        if s in s1_set:
            truth_pairs += len(ms)
    if truth_pairs == 0:
        return {}
    s1_of = s1_ids[pairs["s1"].to_numpy()]
    g_of = g_ids[pairs["g"].to_numpy()]
    hit = np.fromiter((g in gt.get(s, ()) for s, g in zip(s1_of, g_of)), dtype=bool, count=len(pairs))
    rank = pairs["rank"].to_numpy()
    report = {"truth_pairs": truth_pairs, "n_pairs": len(pairs),
              "avg_cands": len(pairs) / max(1, len(s1_set))}
    for k in ks:
        report[f"recall@{k}"] = float((hit & (rank < k)).sum() / truth_pairs)
    return report
