"""Empirical low-rank structure study of DJ sets (selection space).

Tests, with numbers, the repo's low-rank worldview:
  A1  overall rank of the set x recording incidence matrix vs a
      degree-preserving null,
  A2  within-DJ rank vs random same-size subsets of the corpus,
  A3  DJ-basis principal-angle similarity + leave-one-set-out retrieval,
  M   metadata/ML: DJ + set-type identifiability from the representation,
      PC~metadata structure, aligner-difficulty profile by set type.

Representation B (audio features) is gated on coverage and skipped if <50%.

Inputs come from pull_data.py (data/*.tsv). Outputs: printed report +
data/results.json + data/affinity.tsv + data/per_set.tsv.

Usage: venvs/audio/bin/python eda/alignment/low_rank/low_rank_study.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import TruncatedSVD

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RNG = np.random.default_rng(0)

MIN_SLOTS = 10  # a set qualifies with >= this many distinct recordings
MIN_SETS_PER_DJ = 8  # a DJ qualifies with >= this many qualifying sets
GLOBAL_K = 500  # truncation for the global spectrum
BASIS_CAP = 10  # cap on per-DJ basis dimension
LOSO_PER_DJ = 20  # held-out sets sampled per DJ for retrieval
N_CONTROLS = 10  # random same-size control subsets per DJ (A2)

SHOW_RE = re.compile(
    r"\b(radio|podcast|sessions?|show|episode|weekly|mixtape|fm|"
    r"chart|top ?\d+)\b|#\d+|\b\d{2,4}\s*$",
    re.IGNORECASE,
)
B2B_RE = re.compile(r"\bb[2t]b\b", re.IGNORECASE)


# --------------------------------------------------------------------------
# loading + metadata
# --------------------------------------------------------------------------


def classify_set_type(title: str) -> str:
    """1001TL convention: 'Artist @ Venue/Festival' = live gig;
    'Artist - Show Name NNN' / 'Radio' / 'Podcast' = episodic show."""
    if " @ " in title:
        return "live"
    if SHOW_RE.search(title):
        return "show"
    return "other"


def load_per_set() -> pd.DataFrame:
    sets = pd.read_csv(DATA / "sets.tsv", sep="\t", dtype={"set_id": str})
    dj_map = pd.read_csv(DATA / "dj_map.tsv", sep="\t", dtype=str)
    slots = pd.read_csv(DATA / "slots.tsv", sep="\t", dtype=str)

    agg = slots.groupby("set_id").agg(
        n_slot_rows=("recording_id", "size"),
        n_distinct=("recording_id", "nunique"),
        acap_share=("claimed_stem", lambda s: float(np.mean(s == "acappella"))),
        instr_share=("claimed_stem", lambda s: float(np.mean(s == "instrumental"))),
    )
    df = sets.merge(dj_map, on="set_id", how="left").merge(agg, on="set_id", how="left")
    df["dj"] = df["dj"].fillna("unknown")
    df["title"] = df["title"].fillna("")
    df["is_b2b"] = df["title"].str.contains(B2B_RE)
    df["set_type"] = df["title"].map(classify_set_type)
    df["location"] = df["title"].str.extract(r" @ (.+)$", expand=False)
    df["year"] = pd.to_numeric(df["date_played"].astype(str).str[:4], errors="coerce")
    df["qualifies"] = df["n_distinct"].fillna(0) >= MIN_SLOTS
    return df


def build_matrix(
    slots: pd.DataFrame, set_ids: list[str]
) -> tuple[sp.csr_matrix, list[str], list[str]]:
    """Binary set x recording incidence over the given sets, rows L2-normalized
    later by callers. Duplicate (set, recording) slot rows collapse to 1."""
    keep = slots[slots["set_id"].isin(set_ids)]
    pairs = keep[["set_id", "recording_id"]].drop_duplicates()
    row_cat = pd.Categorical(pairs["set_id"], categories=set_ids)
    col_labels = sorted(pairs["recording_id"].unique())
    col_cat = pd.Categorical(pairs["recording_id"], categories=col_labels)
    mat = sp.coo_matrix(
        (np.ones(len(pairs), dtype=np.float64), (row_cat.codes, col_cat.codes)),
        shape=(len(set_ids), len(col_labels)),
    ).tocsr()
    return mat, set_ids, col_labels


def l2_rows(mat: sp.csr_matrix) -> sp.csr_matrix:
    norms = np.sqrt(mat.multiply(mat).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    inv = sp.diags(1.0 / norms)
    return (inv @ mat).tocsr()


# --------------------------------------------------------------------------
# A1: global spectrum vs configuration-model null
# --------------------------------------------------------------------------


def spectrum(mat: sp.csr_matrix, k: int) -> np.ndarray:
    svd = TruncatedSVD(
        n_components=k, algorithm="randomized", n_iter=10, random_state=0
    )
    svd.fit(mat)
    return np.asarray(svd.singular_values_)


def energy_components(
    svals: np.ndarray, total_energy: float, levels: tuple[float, ...] = (0.5, 0.8, 0.9)
) -> dict[str, float | None]:
    cum = np.cumsum(svals**2) / total_energy
    out: dict[str, float | None] = {}
    for lv in levels:
        idx = np.searchsorted(cum, lv) + 1
        out[f"comps_{int(lv * 100)}"] = int(idx) if cum[-1] >= lv else None
    out["captured_at_k"] = float(cum[-1])
    return out


def config_null(mat: sp.csr_matrix, rng: np.random.Generator) -> sp.csr_matrix:
    """Bipartite configuration-model shuffle: keep row- and column-degree
    sequences on the edge list, randomly re-pair endpoints, dedupe collisions.
    At this density (~0.014%) dedup loss is <<1% of edges (reported)."""
    coo = mat.tocoo()
    cols = coo.col.copy()
    rng.shuffle(cols)
    pairs = np.unique(np.stack([coo.row, cols], axis=1), axis=0)
    null = sp.coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=mat.shape
    ).tocsr()
    null.loss = 1.0 - len(pairs) / coo.nnz  # type: ignore[attr-defined]
    return null


# --------------------------------------------------------------------------
# A2/A3 helpers: per-DJ Gram spectra and bases
# --------------------------------------------------------------------------


def full_gram_spectrum(sub: sp.csr_matrix) -> np.ndarray:
    """Exact full squared-singular-value spectrum via the n x n Gram."""
    g = (sub @ sub.T).toarray()
    return np.sort(np.linalg.eigvalsh(g))[::-1]


def r80(eigvals: np.ndarray, total: float) -> int:
    cum = np.cumsum(np.clip(eigvals, 0, None)) / total
    return int(np.searchsorted(cum, 0.8) + 1)


@dataclass(frozen=True)
class DJBasis:
    slug: str
    xn: sp.csr_matrix  # L2-normalized rows over the SHARED vocab
    w: np.ndarray  # n x k, U/s, so V = xn.T @ w is orthonormal
    k: int
    r80_val: int


def topk_basis(xn: sp.csr_matrix, k: int) -> np.ndarray:
    g = (xn @ xn.T).toarray()
    vals, vecs = eigsh(g, k=min(k, g.shape[0] - 1), which="LA")
    order = np.argsort(vals)[::-1]
    vals, vecs = np.clip(vals[order], 1e-12, None), vecs[:, order]
    return vecs / np.sqrt(vals)  # W = U / s


def dj_similarity(a: DJBasis, b: DJBasis) -> float:
    """Mean cosine of principal angles between the two right subspaces."""
    cross = (a.xn @ b.xn.T).toarray()  # n_a x n_b
    m = a.w.T @ cross @ b.w  # k_a x k_b = V_a^T V_b
    svals = np.linalg.svd(m, compute_uv=False)
    return float(np.mean(np.clip(svals[: min(a.k, b.k)], 0, 1)))


def proj_energy(basis: DJBasis, rows: sp.csr_matrix) -> np.ndarray:
    """||V^T x||^2 per row of `rows` (rows must be L2-normalized)."""
    p = basis.w.T @ (basis.xn @ rows.T).toarray()  # k x m
    return np.sum(p**2, axis=0)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> None:  # noqa: C901 - linear study script
    results: dict = {}
    per_set = load_per_set()
    slots = pd.read_csv(DATA / "slots.tsv", sep="\t", dtype=str)
    per_set.to_csv(DATA / "per_set.tsv", sep="\t", index=False)

    cov = pd.read_csv(DATA / "coverage.txt", sep="\t").iloc[0]
    results["rep_b_coverage"] = {
        "slots_with_features_pct": 100.0
        * cov["n_slots_with_features"]
        / cov["n_slots"],
        "skipped": True,
    }

    qual = per_set[per_set["qualifies"]].copy()
    set_ids = qual["set_id"].tolist()
    x, _, vocab = build_matrix(slots, set_ids)
    xn = l2_rows(x)
    n, v = x.shape
    print(
        f"matrix: {n} sets x {v} recordings, nnz={x.nnz} (density {x.nnz / n / v:.5%})"
    )
    results["matrix"] = {"sets": n, "vocab": v, "nnz": int(x.nnz)}

    # ---- A1 ----------------------------------------------------------------
    print("\n[A1] global spectrum (randomized SVD, k=%d) ..." % GLOBAL_K)
    s_real = spectrum(xn, GLOBAL_K)
    a1_real = energy_components(s_real, total_energy=float(n))
    nulls = []
    for i in range(2):
        nm = config_null(x, np.random.default_rng(100 + i))
        print(f"  null {i}: dedup loss {nm.loss:.4%}")
        s_null = spectrum(l2_rows(nm), GLOBAL_K)
        nulls.append(energy_components(s_null, total_energy=float(n)))
    pc1_share = float(s_real[0] ** 2 / n)
    col_freq = np.asarray(x.sum(axis=0)).ravel()
    svd1 = TruncatedSVD(
        n_components=1, algorithm="randomized", n_iter=10, random_state=0
    ).fit(xn)
    v1 = np.abs(svd1.components_[0])
    pc1_pop_corr = float(np.corrcoef(v1, col_freq)[0, 1])
    results["A1"] = {
        "real": a1_real,
        "nulls": nulls,
        "pc1_energy_share": pc1_share,
        "pc1_vs_track_frequency_r": pc1_pop_corr,
        "top_singular_values": s_real[:20].tolist(),
    }
    print(f"  real: {a1_real}")
    for i, nl in enumerate(nulls):
        print(f"  null{i}: {nl}")
    print(f"  PC1 energy {pc1_share:.3%}, corr(|V1|, track freq) = {pc1_pop_corr:.3f}")

    # ---- A2 ----------------------------------------------------------------
    dj_counts = qual[~qual["dj"].isin(["__multi__", "unknown"])].groupby("dj").size()
    djs = sorted(dj_counts[dj_counts >= MIN_SETS_PER_DJ].index)
    print(f"\n[A2] {len(djs)} DJs with >= {MIN_SETS_PER_DJ} qualifying sets")
    row_of = {sid: i for i, sid in enumerate(set_ids)}
    a2_rows = []
    for dj in djs:
        idx = np.array([row_of[s] for s in qual.loc[qual["dj"] == dj, "set_id"]])
        sub = xn[idx]
        sub = sub[:, np.asarray(sub.sum(axis=0)).ravel() > 0]  # own support
        eig = full_gram_spectrum(sub)
        r = r80(eig, float(len(idx)))
        ctrl = []
        for _ in range(N_CONTROLS):
            ridx = RNG.choice(n, size=len(idx), replace=False)
            rsub = xn[ridx]
            rsub = rsub[:, np.asarray(rsub.sum(axis=0)).ravel() > 0]
            ctrl.append(r80(full_gram_spectrum(rsub), float(len(idx))))
        a2_rows.append(
            {
                "dj": dj,
                "n_sets": int(len(idx)),
                "r80": r,
                "r80_norm": r / min(sub.shape),
                "ctrl_r80_mean": float(np.mean(ctrl)),
                "ctrl_r80_sd": float(np.std(ctrl)),
            }
        )
    a2 = pd.DataFrame(a2_rows)
    a2["ratio"] = a2["r80"] / a2["ctrl_r80_mean"]
    results["A2"] = {
        "per_dj": a2_rows,
        "median_r80_norm": float(a2["r80_norm"].median()),
        "median_ratio_vs_control": float(a2["ratio"].median()),
        "frac_djs_below_control": float((a2["ratio"] < 1).mean()),
    }
    print(
        a2[["dj", "n_sets", "r80", "ctrl_r80_mean", "ratio"]]
        .sort_values("ratio")
        .to_string(index=False, max_rows=20)
    )
    print(
        f"  median r80/min(dim) = {a2['r80_norm'].median():.3f}; "
        f"median real/control ratio = {a2['ratio'].median():.3f}; "
        f"{(a2['ratio'] < 1).mean():.1%} of DJs below control"
    )

    # ---- A3 ----------------------------------------------------------------
    print("\n[A3] per-DJ bases + principal angles + LOSO retrieval ...")
    bases: dict[str, DJBasis] = {}
    dj_idx: dict[str, np.ndarray] = {}
    for dj in djs:
        idx = np.array([row_of[s] for s in qual.loc[qual["dj"] == dj, "set_id"]])
        dj_idx[dj] = idx
        r = int(a2.loc[a2["dj"] == dj, "r80"].iloc[0])
        k = min(r, BASIS_CAP, len(idx) - 1)
        sub = xn[idx]
        bases[dj] = DJBasis(dj, sub, topk_basis(sub, k), k, r)

    sim = np.zeros((len(djs), len(djs)))
    for i, a in enumerate(djs):
        for j in range(i, len(djs)):
            sim[i, j] = sim[j, i] = (
                1.0 if i == j else dj_similarity(bases[a], bases[djs[j]])
            )
    sim_df = pd.DataFrame(sim, index=djs, columns=djs)

    named = ["twofriends", "johnsummit", "discolines", "itsmurph"]
    rich = list(dj_counts.sort_values(ascending=False).index[:10])
    table_djs = [d for d in dict.fromkeys(rich + named) if d in bases]
    sim_df.loc[table_djs, table_djs].round(3).to_csv(DATA / "affinity.tsv", sep="\t")

    # LOSO retrieval
    rng = np.random.default_rng(7)
    correct = 0
    total = 0
    per_dj_acc = {}
    for dj in djs:
        idx = dj_idx[dj]
        held = rng.choice(len(idx), size=min(LOSO_PER_DJ, len(idx)), replace=False)
        hits = 0
        for h in held:
            x_h = xn[idx[h]]
            mask = np.ones(len(idx), bool)
            mask[h] = False
            sub = xn[idx[mask]]
            own = DJBasis(
                dj, sub, topk_basis(sub, bases[dj].k), bases[dj].k, bases[dj].r80_val
            )
            scores = {d: float(proj_energy(bases[d], x_h)[0]) for d in djs if d != dj}
            scores[dj] = float(proj_energy(own, x_h)[0])
            if max(scores, key=scores.get) == dj:  # type: ignore[arg-type]
                hits += 1
        per_dj_acc[dj] = hits / len(held)
        correct += hits
        total += len(held)
    loso_acc = correct / total
    results["A3"] = {
        "n_djs": len(djs),
        "chance": 1.0 / len(djs),
        "loso_top1_acc": loso_acc,
        "basis_cap": BASIS_CAP,
        "offdiag_sim_mean": float(sim[np.triu_indices(len(djs), 1)].mean()),
        "offdiag_sim_sd": float(sim[np.triu_indices(len(djs), 1)].std()),
        "per_dj_acc": per_dj_acc,
        "affinity_table_djs": table_djs,
    }
    print(
        f"  LOSO top-1 own-DJ retrieval: {loso_acc:.1%} "
        f"(chance {1 / len(djs):.2%}, {total} held-out sets)"
    )
    print(
        f"  off-diagonal subspace similarity: "
        f"{results['A3']['offdiag_sim_mean']:.3f} "
        f"+/- {results['A3']['offdiag_sim_sd']:.3f}"
    )
    print(sim_df.loc[table_djs, table_djs].round(2).to_string())

    # ---- M: metadata + ML ---------------------------------------------------
    print("\n[M] metadata / ML ...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import NearestCentroid

    svd = TruncatedSVD(
        n_components=200, algorithm="randomized", n_iter=10, random_state=0
    )
    z = svd.fit_transform(xn)
    qual = qual.reset_index(drop=True)

    m_dj = qual["dj"].isin(djs).to_numpy()
    y_dj = qual.loc[m_dj, "dj"].to_numpy()
    clf = LogisticRegression(max_iter=3000, C=1.0)
    acc_dj = cross_val_score(clf, z[m_dj], y_dj, cv=5, n_jobs=-1)
    acc_nc = cross_val_score(NearestCentroid(), z[m_dj], y_dj, cv=5)
    maj = pd.Series(y_dj).value_counts(normalize=True).iloc[0]
    results["M_dj_clf"] = {
        "logreg_acc": float(acc_dj.mean()),
        "logreg_sd": float(acc_dj.std()),
        "nearest_centroid_acc": float(acc_nc.mean()),
        "chance": 1.0 / len(djs),
        "majority_class": float(maj),
        "n": int(m_dj.sum()),
        "n_classes": len(djs),
    }
    print(
        f"  DJ clf: logreg {acc_dj.mean():.1%} / centroid "
        f"{acc_nc.mean():.1%} vs chance {1 / len(djs):.2%} "
        f"(majority {maj:.1%}, n={m_dj.sum()})"
    )

    m_st = qual["set_type"].isin(["live", "show"]).to_numpy()
    y_st = (qual.loc[m_st, "set_type"] == "live").to_numpy()
    auc_st = cross_val_score(
        LogisticRegression(max_iter=3000),
        z[m_st],
        y_st,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )
    pc_auc = [float(roc_auc_score(y_st, z[m_st, i])) for i in range(10)]
    pc_auc = [max(a, 1 - a) for a in pc_auc]
    results["M_settype"] = {
        "counts": qual["set_type"].value_counts().to_dict(),
        "logreg_auc": float(auc_st.mean()),
        "base_rate_live": float(y_st.mean()),
        "pc_auc_top10": pc_auc,
    }
    print(
        f"  set_type (live vs show): AUC {auc_st.mean():.3f} "
        f"(base rate live {y_st.mean():.1%}); per-PC AUC top10: "
        + ", ".join(f"{a:.2f}" for a in pc_auc)
    )

    # Explicit 'b2b' in titles is nearly absent in this corpus (n~3); the
    # usable collaboration proxy is __multi__: the set appears in >= 2 DJs'
    # scrape job files (e.g. "Illenium & Said The Sky @ ...").
    y_b2b = qual["is_b2b"].to_numpy()
    y_multi = (qual["dj"] == "__multi__").to_numpy()
    results["M_b2b"] = {
        "n_b2b_explicit": int(y_b2b.sum()),
        "n_multi_dj": int(y_multi.sum()),
    }
    if y_b2b.sum() >= 50:
        auc_b2b = cross_val_score(
            LogisticRegression(max_iter=3000),
            z,
            y_b2b,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
        )
        results["M_b2b"]["logreg_auc_explicit"] = float(auc_b2b.mean())
    if y_multi.sum() >= 50:
        auc_m = cross_val_score(
            LogisticRegression(max_iter=3000),
            z,
            y_multi,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
        )
        results["M_b2b"]["logreg_auc_multi_dj"] = float(auc_m.mean())
        print(
            f"  b2b explicit n={y_b2b.sum()} (too small for a claim); "
            f"multi-DJ proxy n={y_multi.sum()}, AUC {auc_m.mean():.3f}"
        )

    # feature importance: which components carry DJ vs set_type
    clf_dj = LogisticRegression(max_iter=3000, C=1.0).fit(z[m_dj], y_dj)
    imp_dj = np.abs(clf_dj.coef_).mean(axis=0)
    clf_st = LogisticRegression(max_iter=3000).fit(z[m_st], y_st)
    imp_st = np.abs(clf_st.coef_[0])
    results["M_importance"] = {
        "dj_top10_pcs": (np.argsort(imp_dj)[::-1][:10] + 1).tolist(),
        "settype_top10_pcs": (np.argsort(imp_st)[::-1][:10] + 1).tolist(),
    }
    print(f"  top PCs for DJ: {results['M_importance']['dj_top10_pcs']}")
    print(f"  top PCs for set_type: {results['M_importance']['settype_top10_pcs']}")

    # PC ~ other metadata correlations
    meta_corr = {}
    for name, col in [
        ("year", qual["year"]),
        ("log_views", np.log1p(pd.to_numeric(qual["views"], errors="coerce"))),
        ("n_distinct", qual["n_distinct"]),
    ]:
        ok = col.notna().to_numpy()
        col_ok = pd.Series(col.to_numpy()[ok])  # positional, index-aligned
        meta_corr[name] = [
            float(pd.Series(z[ok, i]).corr(col_ok, method="spearman")) for i in range(5)
        ]
    results["M_pc_meta_spearman_top5"] = meta_corr

    # aligner-difficulty profile
    prof = (
        qual.groupby("set_type")[
            ["n_slot_rows", "n_distinct", "acap_share", "instr_share"]
        ]
        .mean()
        .round(4)
    )
    prof_b2b = (
        qual.assign(is_multi_dj=y_multi)
        .groupby("is_multi_dj")[
            ["n_slot_rows", "n_distinct", "acap_share", "instr_share"]
        ]
        .mean()
        .round(4)
    )
    results["M_difficulty_by_settype"] = prof.to_dict("index")
    results["M_difficulty_by_multi_dj"] = {
        str(k): v for k, v in prof_b2b.to_dict("index").items()
    }
    print("  difficulty by set_type:\n" + prof.to_string())
    print("  difficulty by multi-DJ (b2b proxy):\n" + prof_b2b.to_string())

    (DATA / "results.json").write_text(json.dumps(results, indent=2))
    print("\nwrote data/results.json, data/affinity.tsv, data/per_set.tsv")


if __name__ == "__main__":
    main()
