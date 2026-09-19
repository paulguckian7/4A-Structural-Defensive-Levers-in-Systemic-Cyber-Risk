
# -*- coding: utf-8 -*-
"""
Paper 4A measurement analysis, per-paper layer outside the frozen core
======================================================================

Question (Paper 4A): on the frozen cemt_core substrate, does a fixed-form
relational combination of the IAE axes classify collapse better than an
additive composite of the same axes, and does the additive composite beat
the strongest genuine single axis?

Provenance
  Extracted from Code80 (CyberReynoldsNumberv4.py, 2026-06-10): the inlined
  formula screen reanalysis, the pairwise coupling null, the rho(K) crossing
  table, the identifiability profile and the P3 evidence pack. Nothing here
  simulates. It reads outputs produced by cemt_core and writes analysis.

Alignment changes from Code80 (declared, see CHANGES below)
  C1  IAE, not IPE. Authority (A) replaces Privilege (P) in every formula,
      in ADD, and as a single-axis baseline (S_A replaces S_P). Execution
      Pathway is written X. Paper 3 v3.42 numbers do not carry forward.
  C2  Pre-specified primary metric. F1 = I*A*X*T/(rem*vis) is the primary
      relational metric. The best-of-set relational AUC is still reported
      but labelled as selected, not as the headline.
  C3  Best single axis is computed, not hardcoded. Code79 hardcoded S_P.
  C4  The stale F4 vs F5 headline is removed. F5 is not single-axis.
  C5  Formulas whose inputs are absent from the substrate output are
      reported as NOT EVALUABLE, and formulas with a constant input (for
      example tau_a, since capacity dynamics are disabled in every reported
      cemt_core experiment) are reported as DEGENERATE. Neither is silently
      dropped, so the fate of the pre-specified set stays visible.
  C6  Outputs are labelled P4A and written to <out>/P4A/.
  C7  The manifest records the cemt_core freeze digest and the SHA-256 of
      every input file. Without a digest the manifest marks the results as
      not citable against the frozen substrate.
  C8  Paper 4B material (R_eff, R_T, D_J, Stage 2) is excluded.

Input contract
  Validation CSV(s): one row per scenario, with the axis columns named in
  COLUMN_MAP, an outcome column (trial_collapse_rate), and optionally
  scenario_id, seed, stage. Several files may be passed (held-out seeds).
  OAT sweep CSV (optional): param_name, level, trial_collapse_rate and
  rho_K (or spectral_radius) for the single-scalar adequacy test and the
  identifiability profile.
  COLUMN_MAP defaults are Code80 names with authority_rate for A. Override
  with --column-map map.json to match the actual cemt_core output.

Usage
  python p4a_screen.py --validation v_seed231.csv v_seed543.csv \
      --sweep oat.csv --out runs/p4a_001 --freeze-digest 97dbae47
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# VERSION (bump on every edit)
# ============================================================================
VERSION = 1
CODE_VERSION = f"P4A-screen v{VERSION} (2026-09-19)"

# ============================================================================
# COLUMN MAP: canonical symbol -> substrate column name
# ============================================================================
# Required for the headline ladder: I, A, X, T, rem, vis.
# Optional (used only by formulas that need them): conn, dep, cp, cp_take,
# cp_pool, tau_a.
DEFAULT_COLUMN_MAP: Dict[str, str] = {
    "I":       "interface_rate",
    "A":       "authority_rate",
    "X":       "execution_rate",
    "T":       "threat_capability",
    "rem":     "remediation_capability",
    "vis":     "visibility",
    "conn":    "connection_density",
    "dep":     "dependency_factor",
    "cp":      "avg_control_planes_per_node",
    "cp_take": "control_plane_takeover_rate",
    "cp_pool": "n_control_plane_pool",
    "tau_a":   "tau_a",
    # outcome and bookkeeping
    "outcome": "trial_collapse_rate",
}

HEADLINE_AXES = ["I", "A", "X", "T", "rem", "vis"]
ADD_POS = ["I", "A", "X", "T"]
ADD_NEG = ["rem", "vis"]

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]
HEADLINE_THRESHOLD = 0.15
PRIMARY_RELATIONAL = "F1"


def _safe_div(num, den):
    return num / np.maximum(den, 1e-9)


# ============================================================================
# FORMULA REGISTRY (canonical symbols; pre-specified set, IAE form)
# ============================================================================
# Structure matches the Code80 set one for one, with P -> A (C1).
FORMULAS: Dict[str, dict] = {
    "F1": {"expr": "I*A*X*T / (rem*vis)", "role": "relational",
           "vars": ["I", "A", "X", "T", "rem", "vis"],
           "f": lambda d: _safe_div(d.I * d.A * d.X * d.T, d.rem * d.vis)},
    "F2": {"expr": "I*A*X*T*dep / (rem*vis)", "role": "relational",
           "vars": ["I", "A", "X", "T", "dep", "rem", "vis"],
           "f": lambda d: _safe_div(d.I * d.A * d.X * d.T * d.dep, d.rem * d.vis)},
    "F3": {"expr": "I*A*X*T / (rem*vis*tau_a)", "role": "relational",
           "vars": ["I", "A", "X", "T", "rem", "vis", "tau_a"],
           "f": lambda d: _safe_div(d.I * d.A * d.X * d.T,
                                    d.rem * d.vis * np.maximum(d.tau_a, 0.01))},
    "F4": {"expr": "I*A*X*T*conn / (rem*vis)", "role": "relational",
           "vars": ["I", "A", "X", "T", "conn", "rem", "vis"],
           "f": lambda d: _safe_div(d.I * d.A * d.X * d.T * d.conn, d.rem * d.vis)},
    "F5": {"expr": "I*T / (rem*vis)", "role": "relational",
           "vars": ["I", "T", "rem", "vis"],
           "f": lambda d: _safe_div(d.I * d.T, d.rem * d.vis)},
    "F6": {"expr": "I*A*X*T*(1+ncp*t) / (rem*vis*(1+(cp/(2.5+cp))*(1-t)))",
           "role": "relational",
           "vars": ["I", "A", "X", "T", "rem", "vis", "cp", "cp_take", "cp_pool"],
           "f": lambda d: _safe_div(
               d.I * d.A * d.X * d.T
               * (1.0 + _safe_div(d.cp, d.cp_pool) * d.cp_take),
               d.rem * d.vis * (1.0 + (d.cp / (2.5 + d.cp)) * (1.0 - d.cp_take)))},
    "F7": {"expr": "I*A*X*T*dep*conn / (rem*vis)", "role": "relational",
           "vars": ["I", "A", "X", "T", "dep", "conn", "rem", "vis"],
           "f": lambda d: _safe_div(d.I * d.A * d.X * d.T * d.dep * d.conn,
                                    d.rem * d.vis)},
    "F8": {"expr": "I*A*X*T / (rem*vis*(1-0.5*tau_a))", "role": "relational",
           "vars": ["I", "A", "X", "T", "rem", "vis", "tau_a"],
           "f": lambda d: _safe_div(d.I * d.A * d.X * d.T,
                                    d.rem * d.vis * np.maximum(1.0 - 0.5 * d.tau_a, 0.01))},
    # Genuine single-axis baselines. Defensive axes negated so that a higher
    # score means higher risk throughout.
    "S_I":    {"expr": "I only",    "role": "single", "vars": ["I"],    "f": lambda d: d.I},
    "S_A":    {"expr": "A only",    "role": "single", "vars": ["A"],    "f": lambda d: d.A},
    "S_X":    {"expr": "X only",    "role": "single", "vars": ["X"],    "f": lambda d: d.X},
    "S_T":    {"expr": "T only",    "role": "single", "vars": ["T"],    "f": lambda d: d.T},
    "S_rem":  {"expr": "-rem only", "role": "single", "vars": ["rem"],  "f": lambda d: -d.rem},
    "S_vis":  {"expr": "-vis only", "role": "single", "vars": ["vis"],  "f": lambda d: -d.vis},
    "S_conn": {"expr": "conn only", "role": "single", "vars": ["conn"], "f": lambda d: d.conn},
    "S_dep":  {"expr": "dep only",  "role": "single", "vars": ["dep"],  "f": lambda d: d.dep},
}
ADD_EXPR = "z(I)+z(A)+z(X)+z(T)-z(rem)-z(vis)  [additive, sample-standardised]"


# ============================================================================
# STATISTICS (DeLong per Sun & Xu 2014; unchanged from Code80)
# ============================================================================

def _midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out


def delong_test(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    """Paired DeLong test: AUCs, difference, SE, z, two-sided p."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y)
    y_s = y[order]
    preds = np.vstack([np.asarray(a, float)[order], np.asarray(b, float)[order]])
    m = int(y_s.sum())
    n = len(y_s) - m
    if m == 0 or n == 0:
        return dict(auc_a=np.nan, auc_b=np.nan, diff=np.nan, se=np.nan, z=np.nan, p=np.nan)
    tx = np.array([_midrank(r[:m]) for r in preds])
    ty = np.array([_midrank(r[m:]) for r in preds])
    tz = np.array([_midrank(r) for r in preds])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01) / m + np.cov(v10) / n)
    diff = aucs[0] - aucs[1]
    se = float(np.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 0.0)))
    z = diff / se if se > 0 else 0.0
    return dict(auc_a=float(aucs[0]), auc_b=float(aucs[1]), diff=float(diff),
                se=se, z=float(z), p=float(2.0 * (1.0 - norm.cdf(abs(z)))))


def bootstrap_auc_ci(y, s, n_boot=1000, seed=42) -> Tuple[float, float]:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        try:
            out.append(roc_auc_score(y[idx], s[idx]))
        except ValueError:
            pass
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def bh_adjust(p: List[float]) -> List[float]:
    p = np.asarray(p, float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        k = m - rank
        prev = min(prev, p[idx] * m / k)
        adj[idx] = prev
    return adj.tolist()


def fit_oof_logistic(X, y, n_splits=5, seed=42) -> np.ndarray:
    y = np.asarray(y).astype(int)
    k = min(n_splits, int(y.sum()), int((1 - y).sum()))
    if k < 2:
        return np.full(len(y), float(y.mean()))
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        try:
            lr = LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed)
            lr.fit(sc.transform(X[tr]), y[tr])
            oof[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
        except Exception:
            oof[te] = float(y[tr].mean())
    return oof


def epv_status(epv: float) -> str:
    if epv >= 10:
        return "OK"
    if epv >= 5:
        return "MARGINAL"
    if epv >= 2:
        return "EXPLORATORY"
    return "SUSPECT"


# ============================================================================
# LOADING AND SCORING
# ============================================================================

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_validation(paths: List[str], colmap: Dict[str, str]) -> pd.DataFrame:
    """Read validation CSVs and rename substrate columns to canonical symbols."""
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        if "theory" in d.columns:  # Code80 legacy files; cemt_core has none
            d = d[d["theory"] == "ngm"]
        d = d.copy()
        d["source_file"] = os.path.basename(p)
        if "seed" not in d.columns:
            d["seed"] = os.path.splitext(os.path.basename(p))[0]
        if "stage" not in d.columns:
            d["stage"] = "F"
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    rename = {v: k for k, v in colmap.items() if v in df.columns}
    df = df.rename(columns=rename)
    if "outcome" not in df.columns:
        raise SystemExit(f"Outcome column '{colmap['outcome']}' not found in validation input.")
    missing = [a for a in HEADLINE_AXES if a not in df.columns]
    if missing:
        raise SystemExit(
            "Headline axes missing from validation input: "
            + ", ".join(f"{a} (expected column '{colmap[a]}')" for a in missing)
            + ". Supply --column-map to match the cemt_core output.")
    return df.reset_index(drop=True)


def formula_status(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluable, NOT EVALUABLE (inputs absent) or DEGENERATE (a constant input)."""
    rows = []
    for name, spec in FORMULAS.items():
        absent = [v for v in spec["vars"] if v not in df.columns]
        const = [v for v in spec["vars"] if v in df.columns
                 and float(df[v].std()) <= 1e-12]
        status = "NOT EVALUABLE" if absent else "DEGENERATE" if const else "OK"
        rows.append(dict(formula=name, role=spec["role"], expr=spec["expr"],
                         status=status, absent_inputs=";".join(absent),
                         constant_inputs=";".join(const)))
    rows.append(dict(formula="ADD", role="additive", expr=ADD_EXPR, status="OK",
                     absent_inputs="", constant_inputs=""))
    return pd.DataFrame(rows)


def add_score(df: pd.DataFrame) -> Tuple[np.ndarray, List[dict]]:
    """Sample-standardised additive composite, plus the reference means and SDs."""
    ref = []
    total = np.zeros(len(df))
    for sign, axes in ((+1, ADD_POS), (-1, ADD_NEG)):
        for a in axes:
            x = df[a].values.astype(float)
            mu, sd = float(x.mean()), float(x.std())
            ref.append(dict(axis=a, sign=sign, mean=mu, sd=sd, n=len(x)))
            if sd > 1e-12:
                total += sign * (x - mu) / sd
    return total, ref


def score(df: pd.DataFrame, name: str) -> np.ndarray:
    if name == "ADD":
        return add_score(df)[0]
    # Attribute access on a DataFrame would resolve d.T to the transpose, so
    # formulas receive a namespace of plain arrays instead.
    ns = SimpleNamespace(**{v: df[v].values.astype(float) for v in FORMULAS[name]["vars"]})
    return np.asarray(FORMULAS[name]["f"](ns), float)


def vars_of(name: str) -> List[str]:
    return ADD_POS + ADD_NEG if name == "ADD" else FORMULAS[name]["vars"]


def usable(status: pd.DataFrame) -> List[str]:
    return status.loc[status["status"] == "OK", "formula"].tolist()


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_label(df, y, label, names, n_boot) -> pd.DataFrame:
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    rows = []
    for name in names:
        s = score(df, name)
        v = vars_of(name)
        try:
            auc_s = roc_auc_score(y, s)
        except ValueError:
            auc_s = np.nan
        lo_s, hi_s = bootstrap_auc_ci(y, s, n_boot)
        oof = fit_oof_logistic(df[v].values, y)
        try:
            auc_f = roc_auc_score(y, oof)
        except ValueError:
            auc_f = np.nan
        lo_f, hi_f = bootstrap_auc_ci(y, oof, n_boot)
        epv = n_pos / max(len(v), 1)
        role = "additive" if name == "ADD" else FORMULAS[name]["role"]
        rows.append(dict(label=label, formula=name, role=role, n_vars=len(v),
                         n_pos=n_pos, n_neg=n_neg, epv=epv, epv_status=epv_status(epv),
                         auc_scalar=auc_s, scalar_ci_lo=lo_s, scalar_ci_hi=hi_s,
                         auc_fitted=auc_f, fitted_ci_lo=lo_f, fitted_ci_hi=hi_f,
                         fitted_minus_scalar=auc_f - auc_s))
    return pd.DataFrame(rows)


def pairwise_delong(df, y, label, names, scoring) -> pd.DataFrame:
    pred = {}
    for n in names:
        pred[n] = score(df, n) if scoring == "scalar" else fit_oof_logistic(df[vars_of(n)].values, y)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = delong_test(y, pred[names[i]], pred[names[j]])
            rows.append(dict(label=label, scoring=scoring,
                             pair=f"{names[i]} vs {names[j]}", **r))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_bh"] = bh_adjust(out["p"].tolist())
        out["reject_bh"] = out["p_bh"] <= 0.05
    return out


def _pair_row(dl: pd.DataFrame, label: str, scoring: str, a: str, b: str) -> Optional[pd.Series]:
    s = dl[(dl["label"] == label) & (dl["scoring"] == scoring)]
    for pair, flip in ((f"{a} vs {b}", 1), (f"{b} vs {a}", -1)):
        r = s[s["pair"] == pair]
        if len(r):
            r = r.iloc[0].copy()
            r["diff"] *= flip
            r["z"] *= flip
            return r
    return None


def ladder(aucs: pd.DataFrame, dl: pd.DataFrame, label: str) -> dict:
    """Primary ladder: pre-specified F1, ADD, best genuine single axis."""
    a = aucs[aucs["label"] == label].set_index("formula")
    singles = a[a["role"] == "single"]
    best_single = singles["auc_scalar"].idxmax() if len(singles) else None
    rel = a[a["role"] == "relational"]
    best_rel = rel["auc_scalar"].idxmax() if len(rel) else None
    out = dict(label=label,
               primary=PRIMARY_RELATIONAL,
               primary_auc=float(a.loc[PRIMARY_RELATIONAL, "auc_scalar"])
               if PRIMARY_RELATIONAL in a.index else np.nan,
               add_auc=float(a.loc["ADD", "auc_scalar"]) if "ADD" in a.index else np.nan,
               best_single=best_single,
               best_single_auc=float(a.loc[best_single, "auc_scalar"]) if best_single else np.nan,
               selected_relational=best_rel,
               selected_relational_auc=float(a.loc[best_rel, "auc_scalar"]) if best_rel else np.nan,
               n_pos=int(a["n_pos"].iloc[0]) if len(a) else 0)
    for tag, x, yv in (("primary_vs_add", PRIMARY_RELATIONAL, "ADD"),
                       ("add_vs_single", "ADD", best_single),
                       ("primary_vs_single", PRIMARY_RELATIONAL, best_single)):
        for scoring in ("scalar", "fitted"):
            r = _pair_row(dl, label, scoring, x, yv) if yv else None
            out[f"{tag}_{scoring}_diff"] = float(r["diff"]) if r is not None else np.nan
            out[f"{tag}_{scoring}_z"] = float(r["z"]) if r is not None else np.nan
            out[f"{tag}_{scoring}_p_bh"] = float(r["p_bh"]) if r is not None else np.nan
    out["ordering_holds"] = bool(out["primary_auc"] > out["add_auc"] > out["best_single_auc"])
    return out


def labels_for(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {f"tcr_ge_{t:.2f}": (df["outcome"] >= t).astype(int).values for t in THRESHOLDS}


# ============================================================================
# PAIRWISE COUPLING NULL (ported from Code80, canonical axes)
# ============================================================================

def pairwise_coupling(df, y, label, axes, n_perm=100, seed=12345) -> pd.DataFrame:
    """Additive vs interaction logistic per pair, with a label-permutation null."""
    if int(y.sum()) < 5 or int((1 - y).sum()) < 5:
        return pd.DataFrame()
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    axes = [a for a in axes if a in df.columns and float(df[a].std()) > 1e-12]
    z = {a: (df[a].values - df[a].mean()) / df[a].std() for a in axes}

    def oof_auc(X, yy):
        try:
            p = cross_val_predict(LogisticRegression(max_iter=200), X, yy,
                                  cv=cv, method="predict_proba")[:, 1]
            return roc_auc_score(yy, p)
        except Exception:
            return np.nan

    solo = {a: oof_auc(z[a].reshape(-1, 1), y) for a in axes}
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            a, b = axes[i], axes[j]
            Xa = np.column_stack([z[a], z[b]])
            Xi = np.column_stack([z[a], z[b], z[a] * z[b]])
            add, cpl = oof_auc(Xa, y), oof_auc(Xi, y)
            lift = cpl - add
            null = []
            for _ in range(n_perm):
                yp = rng.permutation(y)
                null.append(oof_auc(Xi, yp) - oof_auc(Xa, yp))
            null = np.asarray([v for v in null if np.isfinite(v)])
            mu = float(null.mean()) if len(null) else np.nan
            sd = float(null.std()) if len(null) else np.nan
            zz = (lift - mu) / sd if sd and sd > 1e-9 else np.nan
            rows.append(dict(label=label, axis_i=a, axis_j=b,
                             single_auc_i=solo[a], single_auc_j=solo[b],
                             additive_pair_auc=add, interaction_pair_auc=cpl,
                             interaction_lift=lift, null_mean=mu, null_sd=sd,
                             z_score=zz, significant=bool(np.isfinite(zz) and zz >= 1.96)))
    out = pd.DataFrame(rows)
    return out.sort_values("z_score", ascending=False) if len(out) else out


# ============================================================================
# SWEEP-BASED TESTS: single-scalar adequacy and identifiability
# ============================================================================

def rho_at_crossing(sweep: pd.DataFrame, threshold=HEADLINE_THRESHOLD) -> pd.DataFrame:
    """rho(K) at the parameter level where the OAT collapse rate crosses threshold.

    Only genuine interpolated crossings are returned as rows with
    crossing_kind == 'interpolated'; parameters that never cross are kept
    and labelled so they cannot enter a headline table by accident.
    """
    rho_col = next((c for c in ("rho_K", "spectral_radius") if c in sweep.columns), None)
    if rho_col is None:
        return pd.DataFrame()
    rows = []
    for param, g in sweep.groupby("param_name"):
        g = g.sort_values("level")
        L, R, H = (g["level"].values.astype(float), g["trial_collapse_rate"].values.astype(float),
                   g[rho_col].values.astype(float))
        kind, lev, rho = "no_crossing", np.nan, np.nan
        for i in range(len(R) - 1):
            if (R[i] - threshold) * (R[i + 1] - threshold) <= 0 and R[i] != R[i + 1]:
                f = (threshold - R[i]) / (R[i + 1] - R[i])
                lev, rho, kind = L[i] + f * (L[i + 1] - L[i]), H[i] + f * (H[i + 1] - H[i]), "interpolated"
                break
        rows.append(dict(parameter=param, crossing_level=lev, rho_K_at_crossing=rho,
                         crossing_kind=kind, rate_min=float(R.min()), rate_max=float(R.max())))
    return pd.DataFrame(rows)


def identifiability(sweep: pd.DataFrame, n_grid=20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pearson correlation between normalised OAT collapse-rate profiles."""
    prof = {}
    for param, g in sweep.groupby("param_name"):
        v = g.sort_values("level")["trial_collapse_rate"].values.astype(float)
        if len(v) < 2 or np.ptp(v) < 1e-9:
            continue
        v = (v - v.min()) / np.ptp(v)
        prof[param] = np.interp(np.linspace(0, 1, n_grid), np.linspace(0, 1, len(v)), v)
    names = sorted(prof)
    if len(names) < 2:
        return pd.DataFrame(), pd.DataFrame()
    M = np.corrcoef(np.vstack([prof[n] for n in names]))
    pairs = [dict(param_i=names[i], param_j=names[j], profile_corr=float(M[i, j]),
                  abs_corr=float(abs(M[i, j])), flag=("HIGH" if abs(M[i, j]) > 0.95 else
                                                      "MODERATE" if abs(M[i, j]) > 0.85 else "LOW"))
             for i in range(len(names)) for j in range(i + 1, len(names))]
    return (pd.DataFrame(pairs).sort_values("abs_corr", ascending=False),
            pd.DataFrame(M, index=names, columns=names))


# ============================================================================
# FIGURE AND REPORT
# ============================================================================

def plot_roc(df, y, best_single, path, label):
    order = [n for n in (best_single, "ADD", PRIMARY_RELATIONAL) if n]
    colours = {"ADD": "#60A5FA", PRIMARY_RELATIONAL: "#1D4ED8"}
    pts = []
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for n in order:
        s = score(df, n)
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, lw=1.8, color=colours.get(n, "#9CA3AF"),
                label=f"{n} (AUC {roc_auc_score(y, s):.3f})")
        pts += [dict(metric=n, fpr=a, tpr=b) for a, b in zip(fpr, tpr)]
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#999999")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    pd.DataFrame(pts).to_csv(os.path.splitext(path)[0] + "_points.csv", index=False)


def write_summary(path, status, head, robustness, per_seed, digest):
    L = [f"# Paper 4A formula screen, {CODE_VERSION}", ""]
    if not digest:
        L += ["> Substrate freeze digest not recorded. These results are not citable "
              "against the frozen cemt_core substrate.", ""]
    else:
        L += [f"Substrate: cemt_core frozen set digest `{digest}`", ""]
    L += ["## Pre-specified candidate set", "",
          "| Formula | Role | Status | Absent | Constant |", "|---|---|---|---|---|"]
    for _, r in status.iterrows():
        L.append(f"| {r.formula} | {r.role} | {r.status} | {r.absent_inputs or '-'} | "
                 f"{r.constant_inputs or '-'} |")
    h = head
    L += ["", f"## Headline ladder (seed {h.get('seed', '?')}, label `{h['label']}`, n_pos={h['n_pos']})", "",
          f"- Primary relational {h['primary']}: scalar AUC {h['primary_auc']:.3f}",
          f"- Additive ADD: scalar AUC {h['add_auc']:.3f}",
          f"- Best genuine single axis {h['best_single']}: scalar AUC {h['best_single_auc']:.3f}",
          f"- Ordering {h['primary']} > ADD > {h['best_single']}: "
          f"{'holds' if h['ordering_holds'] else 'DOES NOT HOLD'}",
          f"- {h['primary']} vs ADD, scalar: diff {h['primary_vs_add_scalar_diff']:+.3f}, "
          f"z {h['primary_vs_add_scalar_z']:+.2f}, BH p {h['primary_vs_add_scalar_p_bh']:.3g}",
          f"- {h['primary']} vs ADD, fitted: diff {h['primary_vs_add_fitted_diff']:+.3f}, "
          f"BH p {h['primary_vs_add_fitted_p_bh']:.3g} (the fixed-form boundary test)",
          f"- ADD vs {h['best_single']}, scalar: diff {h['add_vs_single_scalar_diff']:+.3f}, "
          f"BH p {h['add_vs_single_scalar_p_bh']:.3g}",
          f"- Selected (best-of-set) relational, reported for completeness only: "
          f"{h['selected_relational']} ({h['selected_relational_auc']:.3f})", "",
          "## Threshold robustness (headline seed)", "",
          "| Stage | Label | F1 | ADD | Best single | n_pos | Holds |", "|---|---|---|---|---|---|---|"]
    for _, r in robustness.iterrows():
        L.append(f"| {r.stage} | {r.label} | {r.primary_auc:.3f} | {r.add_auc:.3f} | "
                 f"{r.best_single} {r.best_single_auc:.3f} | {r.n_pos} | {r.ordering_holds} |")
    if per_seed is not None and len(per_seed):
        L += ["", "## Per-seed confirmation (headline label)", "",
              "| Seed | Stage | F1 | ADD | Best single | Holds |", "|---|---|---|---|---|---|"]
        for _, r in per_seed.iterrows():
            L.append(f"| {r.seed} | {r.stage} | {r.primary_auc:.3f} | {r.add_auc:.3f} | "
                     f"{r.best_single} {r.best_single_auc:.3f} | {r.ordering_holds} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# ============================================================================
# DRIVER
# ============================================================================

def run(validation: List[str], out: str, sweep: Optional[str] = None,
        colmap: Optional[Dict[str, str]] = None, digest: Optional[str] = None,
        n_boot: int = 1000, n_perm: int = 100, headline_seed=None) -> dict:
    colmap = {**DEFAULT_COLUMN_MAP, **(colmap or {})}
    outdir = os.path.join(out, "P4A")
    os.makedirs(outdir, exist_ok=True)
    p = lambda name: os.path.join(outdir, f"P4A_{name}")

    df = load_validation(validation, colmap)
    status = formula_status(df)
    status.to_csv(p("candidate_status.csv"), index=False)
    names = usable(status)
    print(f"{CODE_VERSION}: {len(df)} scenarios, {len(names)} evaluable candidates "
          f"({', '.join(status.loc[status.status != 'OK', 'formula']) or 'none'} excluded)")
    if PRIMARY_RELATIONAL not in names:
        raise SystemExit("Primary relational metric F1 is not evaluable; the headline "
                         "ladder cannot be computed.")

    # The headline and robustness tables use one seed; every other seed is
    # held out and appears only in the per-seed confirmation table.
    hseed = headline_seed if headline_seed is not None else df["seed"].iloc[0]
    dfh = df[df["seed"].astype(str) == str(hseed)].reset_index(drop=True)
    if dfh.empty:
        raise SystemExit(f"Headline seed {hseed} not found in validation input.")
    print(f"Headline seed {hseed}; held-out seeds: "
          f"{sorted(set(df['seed'].astype(str)) - {str(hseed)}) or 'none'}")

    auc_frames, dl_frames, rob_rows, pair_frames = [], [], [], []
    for stage, sdf in dfh.groupby("stage"):
        sdf = sdf.reset_index(drop=True)
        for label, y in labels_for(sdf).items():
            if int(y.sum()) < 3 or int((1 - y).sum()) < 3:
                continue
            a = evaluate_label(sdf, y, label, names, n_boot)
            a.insert(0, "stage", stage)
            d = pd.concat([pairwise_delong(sdf, y, label, names, s) for s in ("scalar", "fitted")],
                          ignore_index=True)
            d.insert(0, "stage", stage)
            auc_frames.append(a)
            dl_frames.append(d)
            rob_rows.append(dict(stage=stage, **ladder(a, d, label)))
        yh = labels_for(sdf)[f"tcr_ge_{HEADLINE_THRESHOLD:.2f}"]
        pc = pairwise_coupling(sdf, yh, f"tcr_ge_{HEADLINE_THRESHOLD:.2f}",
                               [k for k in DEFAULT_COLUMN_MAP if k != "outcome"], n_perm)
        if len(pc):
            pc.insert(0, "stage", stage)
            pair_frames.append(pc)

    aucs = pd.concat(auc_frames, ignore_index=True)
    dls = pd.concat(dl_frames, ignore_index=True)
    rob = pd.DataFrame(rob_rows)
    aucs.to_csv(p("formula_aucs.csv"), index=False)
    dls.to_csv(p("delong_pairs.csv"), index=False)
    rob.to_csv(p("threshold_robustness.csv"), index=False)
    if pair_frames:
        pd.concat(pair_frames, ignore_index=True).to_csv(p("pairwise_coupling.csv"), index=False)

    # Headline: F stage if present, else the first stage.
    hl = f"tcr_ge_{HEADLINE_THRESHOLD:.2f}"
    stages = rob["stage"].unique().tolist()
    hstage = "F" if "F" in stages else stages[0]
    head = rob[(rob.stage == hstage) & (rob.label == hl)].iloc[0].to_dict()
    head["seed"] = str(hseed)
    hdf = dfh[dfh.stage == hstage].reset_index(drop=True)
    plot_roc(hdf, labels_for(hdf)[hl], head["best_single"], p("roc_F1_ADD_single.png"), hl)
    pd.DataFrame(add_score(hdf)[1]).to_csv(p("ADD_standardisation.csv"), index=False)

    # Per-seed confirmation at the headline label.
    seed_rows = []
    if df["seed"].nunique() > 1:
        for (seed, stage), sdf in df.groupby(["seed", "stage"]):
            sdf = sdf.reset_index(drop=True)
            y = labels_for(sdf)[hl]
            if int(y.sum()) < 3 or int((1 - y).sum()) < 3:
                continue
            a = evaluate_label(sdf, y, hl, names, n_boot=200)
            d = pairwise_delong(sdf, y, hl, names, "scalar")
            seed_rows.append(dict(seed=seed, stage=stage, **ladder(a, d, hl)))
    per_seed = pd.DataFrame(seed_rows)
    if len(per_seed):
        per_seed.to_csv(p("per_seed_confirmation.csv"), index=False)

    # Sweep-based tests.
    if sweep:
        sw = pd.read_csv(sweep)
        rc = rho_at_crossing(sw)
        if len(rc):
            rc.to_csv(p("rho_at_crossing_all.csv"), index=False)
            rc[rc.crossing_kind == "interpolated"].to_csv(p("rho_at_crossing.csv"), index=False)
        pairs, mat = identifiability(sw)
        if len(pairs):
            pairs.to_csv(p("identifiability_pairs.csv"), index=False)
            mat.to_csv(p("identifiability_matrix.csv"))

    write_summary(p("summary.md"), status, head, rob, per_seed, digest)

    inputs = list(validation) + ([sweep] if sweep else [])
    manifest = dict(
        code_version=CODE_VERSION,
        generated=_dt.datetime.now().isoformat(timespec="seconds"),
        substrate="cemt_core (frozen)",
        freeze_digest=digest or None,
        citable=bool(digest),
        python=sys.version.split()[0],
        column_map=colmap,
        inputs={os.path.basename(f): _sha256(f) for f in inputs},
        primary_relational=PRIMARY_RELATIONAL,
        headline_label=hl, headline_stage=hstage, headline_seed=str(hseed),
        thresholds=THRESHOLDS,
        excluded=("Paper 4B threshold quantities (R_eff, R_T, D_J, Stage 2) are "
                  "outside this module."),
        files=sorted(os.listdir(outdir)) + ["P4A_manifest.json"],
    )
    with open(p("manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"Headline ({hstage}, {hl}): F1 {head['primary_auc']:.3f} > ADD {head['add_auc']:.3f} "
          f"> {head['best_single']} {head['best_single_auc']:.3f}: "
          f"{'holds' if head['ordering_holds'] else 'does NOT hold'}")
    print(f"Outputs -> {outdir}")
    return dict(head=head, robustness=rob, status=status)


def main(argv=None):
    ap = argparse.ArgumentParser(description=CODE_VERSION)
    ap.add_argument("--validation", nargs="+", required=True)
    ap.add_argument("--sweep")
    ap.add_argument("--out", required=True)
    ap.add_argument("--column-map", help="JSON mapping canonical symbol -> column name")
    ap.add_argument("--freeze-digest", help="cemt_core frozen-set digest")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--headline-seed", help="seed for the headline tables (default: first seed read)")
    a = ap.parse_args(argv)
    cm = json.load(open(a.column_map)) if a.column_map else None
    run(a.validation, a.out, a.sweep, cm, a.freeze_digest, a.n_boot, a.n_perm, a.headline_seed)


if __name__ == "__main__":
    main()