# -*- coding: utf-8 -*-
"""
PhDPaper4A: Paper 4A experiment driver and measurement analysis, one file
=========================================================================

Paper 4A asks which measurements of a cyber system predict its collapse
behaviour on the frozen cemt_core substrate. Everything is in this file,
including the frozen cemt_core v0.8 model, embedded byte for byte and
hash-checked at load. Nothing is imported from the cemt_core repository.
The driver varies scenario conditions only, never the embedded mechanics.

Confirmatory hypotheses as implemented (check the wording against the draft)
  H1   additive vs single axis. The fixed additive composite ADD beats EACH of
       its six constituent single axes (I, A, X, T, rem, vis).
  H2   multiplicative vs additive. F1 = I*A*X*T/(rem*vis) beats ADD.
  H2c  realised conjunction vs product. FC = Cap*T/(rem*vis) beats F1, where
       Cap is the fraction of estate nodes on which I, X and A all hold,
       counting supply through relations (Paper 1B: conferring planes supply
       A, connections and directing planes supply I), rather than the product
       of marginal rates. The node-level co-location measure C_IAX (Paper 1A
       instances only) is reported alongside as FCn, descriptively.
  H3   relational structure. A model on node-level measures plus realised
       relation-class quantities plus relation operators (Fan-out, Cut by
       class, governance crossing) beats the same estimator on node-level
       measures plus conjunction.
  H4   STA completeness. A model on full-STA measures beats each model that
       lacks one dimension (Phi_ST, Phi_SA, Phi_TA), all trained on full-STA
       outcomes and evaluated in a held-out regime (open boundary).

Decision rule (pre-specified, identical for every confirmatory comparison)
  Outcome   the scenario collapse rate itself (fraction of trials meeting the
            frozen sustained-collapse rule); no scenario-level threshold.
  Entry     penetration is assumed, compromise is not (decision 2026-09-20).
            Each trial starts with the attacker at a uniformly chosen
            Interface node; that node is compromised only if I, X and A hold
            there structurally and the gate x_prob * a_prob * T passes
            (entry_certain stays False). The conditional rate, collapse
            given entry compromise, is recorded from the same trials and
            reported descriptively only.
  Measure   Harrell's concordance C between score and collapse rate.
  Pass      Delta C >= MIN_DELTA_C and BH-adjusted p < ALPHA, where p is a
            two-sided paired bootstrap over scenarios and the BH family is
            every confirmatory comparison in H1 to H4 together. H1 and H4
            pass only if every one of their comparisons passes.
  Samples   development: closed boundary, headline seed. Confirmation: closed
            boundary, independent-sample replication seeds (new scenarios,
            not re-seeded trials). Held-out regime: open boundary.
            ADD standardisation and every fitted model are fixed on the
            development sample and applied unchanged elsewhere.
  Secondary the collapse-rate >= 0.15 label is kept only as a stated
            operational definition, for the AUC screen and DeLong tests.

Scenario conditions set by the driver (decision 1, 2026-09-20)
  The frozen generator sets Execution Pathway and Authority present on every
  node and creates no conferring control planes, so conjunction could not be
  varied independently of the marginal product, Cut for X and A was always
  zero, and Authority could never be relationally supplied. The driver
  therefore sets per-node I, X and A presence with a Gaussian-copula
  co-location parameter (marginal rates preserved) and adds conferring
  control planes, some with external issuers. Adaptation is sampled as a
  policy composition: remediation capability, whether trust revocation is
  enabled, and the policy threshold theta. These are spec-level scenario
  conditions; the model mechanics are untouched. The rates authority_rate and
  execution_rate remain the per-node gating probabilities given presence.

Run from Visual Studio with F5 (RUN_CONFIG below), or from a prompt:
  python PhDPaper4A.py --mode smoke
  python PhDPaper4A.py --mode analyse --run-id <existing run id>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# VERSION (bump on every edit)
# ============================================================================
VERSION = 8
CODE_VERSION = f"PhDPaper4A v{VERSION} (2026-09-20)"

# >>> RUN_CONFIG (excluded from the analysis-code hash)
RUN_CONFIG = dict(
    # "ask"      show a menu at start-up (default)
    # "smoke"    small end-to-end run on the pilot seeds, minutes, not for the paper
    # "all"      generate every confirmatory sample and the OAT sweep, then analyse
    # "generate" generate only (resumable)
    # "analyse"  analyse an existing run directory
    mode="ask",
    out_root=r"C:\Users\Paul.Guckian\Documents\Phd\P4A",
    run_id=None,               # None = new timestamped run; set an id to resume or analyse
    workers=None,              # None = cpu_count - 2
    exploratory=False,         # True allows analysis despite a pre-specification
                               # mismatch; the output is then marked not citable
)
# <<< RUN_CONFIG

# ============================================================================
# PRE-SPECIFICATION (everything below is recorded and enforced)
# ============================================================================
DESIGN = dict(
    # Confirmatory seeds. Never used by any pilot or smoke run: the v5 and v6
    # pilots (seeds 231, 543, 311) informed design decisions, so their
    # scenarios are kept out of every confirmatory sample.
    dev_seed=4101,                  # closed boundary, development
    replication_seeds=[4201, 4202], # closed boundary, independent-sample replication
    open_seed=4301,                 # open boundary, held-out regime (H4)
    n_scenarios=1000,               # per seed
    n_trials=500,
    oat=True, oat_levels=9, oat_trials=200,
    min_delta_c=0.02,
    alpha=0.05,
    n_boot=2000,
    boot_seed=20260920,
    model_seed=20260920,
    secondary_threshold=0.15,
    secondary_thresholds=[0.10, 0.15, 0.20, 0.25, 0.30],
    n_perm=1000,
)
# Smoke and pilot runs use the pilot seeds only, never the confirmatory ones.
SMOKE_DESIGN = dict(dev_seed=231, replication_seeds=[543], open_seed=311,
                    n_scenarios=40, n_trials=20,
                    oat_levels=3, oat_trials=10, n_boot=100, n_perm=20)

# (name, low, high, scale, target)
#   scale : lin | log | int | logint
#   target: gen (generate_spec), rate (RateSpec), adapt, time, cond (driver
#           scenario condition applied after generation)
# Not sampled because inert in the frozen layers: RateSpec.synchrony,
# cp_scaling_factor, authority_boost (layer 2 uses min(1, authority_boost)).
SAMPLED: List[Tuple[str, float, float, str, str]] = [
    # IAE presence and co-location (driver conditions)
    ("interface_rate",              0.10, 0.90, "lin",    "cond"),
    ("x_presence_rate",             0.10, 0.90, "lin",    "cond"),
    ("a_presence_rate",             0.10, 0.90, "lin",    "cond"),
    ("colocation",                  0.00, 0.90, "lin",    "cond"),
    # conferring control planes (driver conditions)
    ("n_conferring_planes",         0,    6,    "int",    "cond"),
    ("conferring_span",             0.02, 0.40, "lin",    "cond"),
    ("frac_external_issuers",       0.00, 0.60, "lin",    "cond"),
    # threat and gating (payload side, outside the triad)
    ("threat_capability",           0.50, 3.00, "log",    "rate"),
    ("authority_rate",              0.10, 0.90, "lin",    "rate"),
    ("execution_rate",              0.10, 0.90, "lin",    "rate"),
    # defence (policy composition, Paper 3A fixed policy)
    ("remediation_capability",      0.30, 1.00, "lin",    "adapt"),
    ("revoke_enabled",              0,    1,    "int",    "adapt"),
    ("policy_threshold",            0.00, 0.20, "lin",    "adapt"),
    ("visibility",                  0.30, 1.00, "lin",    "gen"),
    # structure knobs of the frozen generator
    ("avg_internal_zones",          0.50, 2.00, "lin",    "gen"),
    ("dependency_factor",           0.50, 2.00, "lin",    "gen"),
    ("avg_channels_per_node",       0.00, 15.0, "lin",    "gen"),
    ("n_channel_pool",              10,   50,   "logint", "gen"),
    ("frac_external_channel_roots", 0.00, 0.60, "lin",    "gen"),
    ("avg_control_planes_per_node", 0.00, 5.00, "lin",    "gen"),
    ("n_control_plane_pool",        2,    10,   "logint", "gen"),
    ("global_cp_span",              0.30, 1.00, "lin",    "gen"),
    # rates
    ("control_plane_takeover_rate", 0.01, 0.95, "lin",    "rate"),
    ("beta_conn",                   0.05, 0.30, "lin",    "rate"),
    ("control_variance",            0.00, 0.50, "lin",    "rate"),
    ("execution_drift_boost",       0.00, 1.00, "lin",    "rate"),
    # time
    ("latency_steps",               0,    20,   "int",    "time"),
    ("drift_increment",             0.00, 1.00, "lin",    "time"),
]

# Open-boundary regime only: exogenous compromise of external nodes.
OPEN_SAMPLED = [
    ("phantom_cp_rate",             0.01, 0.10, "lin",    "rate"),
    ("phantom_channel_rate",        0.01, 0.10, "lin",    "rate"),
]

FIXED = dict(
    node_count=500, n_zones=80,
    synchrony=0.5, simple_policy=True,                  # Paper 3A fixed policy
    tau_a=0.3,                                          # inert under simple_policy
    max_steps=50, exfil_dwell_steps=5, ablation="STA",
    entry_certain=False,        # penetration assumed, compromise not
    conditional_min_entries=5,  # min entry compromises for the conditional rate
    closed_phantom_rates=0.0,
)

BASELINE = dict(
    interface_rate=0.75, x_presence_rate=0.75, a_presence_rate=0.75, colocation=0.3,
    n_conferring_planes=2, conferring_span=0.1, frac_external_issuers=0.2,
    threat_capability=1.0, authority_rate=0.55, execution_rate=0.55,
    remediation_capability=0.9, revoke_enabled=0, policy_threshold=0.0, visibility=0.85,
    avg_internal_zones=1.0, dependency_factor=2.0, avg_channels_per_node=3.0,
    n_channel_pool=25, frac_external_channel_roots=0.3,
    avg_control_planes_per_node=2.5, n_control_plane_pool=5, global_cp_span=0.3,
    control_plane_takeover_rate=0.08, beta_conn=0.12, control_variance=0.1,
    execution_drift_boost=0.8, latency_steps=2, drift_increment=0.08,
)
OAT_SEED = 999_001

# ---- measurement groups (all computed before any trial is run) -------------
EXOG = ["threat_capability", "authority_rate", "execution_rate", "beta_conn",
        "control_variance", "control_plane_takeover_rate"]
S_NODE = ["I_real", "X_real", "A_real"]
S_CONJ = ["C_IAX", "capability_supplied"]
S_REL = ["conn_density_realised", "channel_per_node", "directing_per_node",
         "conferring_per_node"]
S_OP = ["fanout_max_total", "fanout_mean_directing",
        "fanout_max_directing", "fanout_max_conferring", "fanout_max_sos",
        "cut_I_per_node", "cut_X_per_node", "cut_A_per_node",
        "cut_connection_per_node", "cut_channel_per_node", "cut_directing_per_node",
        "cut_conferring_per_node", "cut_sos_per_node", "cut_max_single_supplier",
        "sos_row_fraction"]
T_F = ["latency_steps", "drift_increment", "execution_drift_boost"]
A_F = ["remediation_capability", "vis_real", "revoke_enabled", "policy_threshold"]
S_ALL = S_NODE + S_CONJ + S_REL + S_OP

FEATURE_SETS = {
    # H3 nested structure sets (T, A and exogenous held in every set)
    "Phi_node": EXOG + T_F + A_F + S_NODE,
    "Phi_conj": EXOG + T_F + A_F + S_NODE + S_CONJ,
    "Phi_rel":  EXOG + T_F + A_F + S_NODE + S_CONJ + S_REL,
    "Phi_op":   EXOG + T_F + A_F + S_ALL,
    # H4 STA sets
    "Phi_STA":  EXOG + S_ALL + T_F + A_F,
    "Phi_ST":   EXOG + S_ALL + T_F,
    "Phi_SA":   EXOG + S_ALL + A_F,
    "Phi_TA":   EXOG + T_F + A_F,
}
ESTIMATOR = dict(kind="HistGradientBoostingRegressor on logit(collapse rate)",
                 max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                 min_samples_leaf=20, l2_regularization=1.0)

# ---- fixed-form metrics (canonical symbol -> column) -----------------------
COLUMN_MAP: Dict[str, str] = {
    "I": "I_real", "A": "A_real", "X": "X_real", "T": "threat_capability",
    "rem": "remediation_capability", "vis": "vis_real",
    "C": "capability_supplied", "Cn": "C_IAX",
    "conn": "conn_density_realised", "dep": "channel_per_node",
    "cp": "directing_per_node", "cp_take": "control_plane_takeover_rate",
    "cp_pool": "n_control_plane_pool", "tau_a": "tau_a",
    "outcome": "trial_collapse_rate",
}
H1_AXES = ["I", "A", "X", "T", "rem", "vis"]
ADD_POS = ["I", "A", "X", "T"]
ADD_NEG = ["rem", "vis"]
COUPLING_AXES = ["I", "A", "X", "T", "rem", "vis", "C", "conn", "dep"]
DESCRIPTIVE_SCORES = ["ADD", "F1", "FC", "FCn", "S_A", "S_C", "S_Cn"]


def _safe_div(num, den):
    return num / np.maximum(den, 1e-9)


FORMULAS: Dict[str, dict] = {
    # multiplicative composites (IAE form of the Code80 set)
    "F1": {"expr": "I*A*X*T/(rem*vis)", "role": "multiplicative",
           "vars": ["I", "A", "X", "T", "rem", "vis"],
           "f": lambda d: _safe_div(d.I * d.A * d.X * d.T, d.rem * d.vis)},
    "FC": {"expr": "Cap*T/(rem*vis), Cap = supplied I^X^A", "role": "conjunction",
           "vars": ["C", "T", "rem", "vis"],
           "f": lambda d: _safe_div(d.C * d.T, d.rem * d.vis)},
    "FCn": {"expr": "C_IAX*T/(rem*vis), node-level only", "role": "conjunction (descriptive)",
            "vars": ["Cn", "T", "rem", "vis"],
            "f": lambda d: _safe_div(d.Cn * d.T, d.rem * d.vis)},
    "F2": {"expr": "I*A*X*T*dep/(rem*vis)", "role": "multiplicative",
           "vars": ["I", "A", "X", "T", "dep", "rem", "vis"],
           "f": lambda d: _safe_div(d.I * d.A * d.X * d.T * d.dep, d.rem * d.vis)},
    "F3": {"expr": "I*A*X*T/(rem*vis*tau_a)", "role": "multiplicative",
           "vars": ["I", "A", "X", "T", "rem", "vis", "tau_a"],
           "f": lambda d: _safe_div(d.I * d.A * d.X * d.T,
                                    d.rem * d.vis * np.maximum(d.tau_a, 0.01))},
    "F4": {"expr": "I*A*X*T*conn/(rem*vis)", "role": "multiplicative",
           "vars": ["I", "A", "X", "T", "conn", "rem", "vis"],
           "f": lambda d: _safe_div(d.I * d.A * d.X * d.T * d.conn, d.rem * d.vis)},
    "F5": {"expr": "I*T/(rem*vis)", "role": "multiplicative",
           "vars": ["I", "T", "rem", "vis"],
           "f": lambda d: _safe_div(d.I * d.T, d.rem * d.vis)},
    "F6": {"expr": "I*A*X*T*(1+cp/cp_pool*cp_take)/(rem*vis*(1+cp/(2.5+cp)*(1-cp_take)))",
           "role": "multiplicative",
           "vars": ["I", "A", "X", "T", "rem", "vis", "cp", "cp_take", "cp_pool"],
           "f": lambda d: _safe_div(
               d.I * d.A * d.X * d.T * (1.0 + _safe_div(d.cp, d.cp_pool) * d.cp_take),
               d.rem * d.vis * (1.0 + (d.cp / (2.5 + d.cp)) * (1.0 - d.cp_take)))},
    "F7": {"expr": "I*A*X*T*dep*conn/(rem*vis)", "role": "multiplicative",
           "vars": ["I", "A", "X", "T", "dep", "conn", "rem", "vis"],
           "f": lambda d: _safe_div(d.I * d.A * d.X * d.T * d.dep * d.conn, d.rem * d.vis)},
    "F8": {"expr": "I*A*X*T/(rem*vis*(1-0.5*tau_a))", "role": "multiplicative",
           "vars": ["I", "A", "X", "T", "rem", "vis", "tau_a"],
           "f": lambda d: _safe_div(d.I * d.A * d.X * d.T,
                                    d.rem * d.vis * np.maximum(1.0 - 0.5 * d.tau_a, 0.01))},
    # H1 constituent single axes (defensive axes negated: higher = riskier)
    "S_I":   {"expr": "I",    "role": "single", "vars": ["I"],   "f": lambda d: d.I},
    "S_A":   {"expr": "A",    "role": "single", "vars": ["A"],   "f": lambda d: d.A},
    "S_X":   {"expr": "X",    "role": "single", "vars": ["X"],   "f": lambda d: d.X},
    "S_T":   {"expr": "T",    "role": "single", "vars": ["T"],   "f": lambda d: d.T},
    "S_rem": {"expr": "-rem", "role": "single", "vars": ["rem"], "f": lambda d: -d.rem},
    "S_vis": {"expr": "-vis", "role": "single", "vars": ["vis"], "f": lambda d: -d.vis},
    # structural single-variable baselines (reported, not H1 comparators)
    "S_C":    {"expr": "Cap",   "role": "structural single", "vars": ["C"],    "f": lambda d: d.C},
    "S_Cn":   {"expr": "C_IAX", "role": "structural single", "vars": ["Cn"],   "f": lambda d: d.Cn},
    "S_conn": {"expr": "conn",  "role": "structural single", "vars": ["conn"], "f": lambda d: d.conn},
    "S_dep":  {"expr": "dep",   "role": "structural single", "vars": ["dep"],  "f": lambda d: d.dep},
}
ADD_EXPR = "z(I)+z(A)+z(X)+z(T)-z(rem)-z(vis), z fixed on the development sample"


# ############################################################################
# EMBEDDED FROZEN CORE: cemt_core v0.8 (no external import)
# ############################################################################
# The six model files of cemt_core v0.8 are carried below byte for byte,
# zlib-compressed and base64-encoded so the text cannot be altered by editing
# or line-ending conversion. At load time each file is decoded, its SHA-256
# is checked against the hash in the v0.8 freeze record (set digest
# 97dbae47...), and it is executed as a module of an in-memory package named
# cemt_core. Nothing is read from disk. If any hash fails, the run is marked
# not citable against the frozen substrate.
#
# To move to a later freeze, regenerate this block from the new tree and
# update EMBEDDED_CORE_VERSION, EMBEDDED_SET_DIGEST and EMBEDDED_HASHES.

EMBEDDED_CORE_VERSION = "0.8"
EMBEDDED_SET_DIGEST = "97dbae4714b278aa322588f69291cf10637fc1aaeedd434eb7590d9edbea3a09"
EMBEDDED_HASHES = {
    "spec": "4974b27b3a40166707878e31ea3b85e455ca0611738b80b86aea94d6f47576c9",
    "relations": "8757d456ad184634d635264dd7b2439a254353a442e7b5d3c1ffdb82ac32860e",
    "network": "49539f8a4670547ccea64cc389ba4fd54bd3c2c28ba2e6856fc583d5bb7172b4",
    "layers": "a0721d50c60de60fea958b796e412e4b89f839aa74321516029f62c342f581a4",
    "step": "6520b76ac28c6de6b98ff7ccf0e3379eb80dfc98d59d0b6460e0dbe7e423c7a6",
    "__init__": "68fb08d9ed682904cdf8614f7dce8153480b7da8b9d324dd578ca56cc896a3df",
}
EMBEDDED_SOURCES = {
    "spec": (
        "eNq1W91y2ziWvtdTYJWLFWdkJnGSrV6lPDVqW5l2dWK7LHV3ulwphiYhiW2K1BCUbY3Le71vMu81"
        "T7LfOQABUJJ/stOtqk7LJHBwcH6/cwB1u93O4ejTRKhEFnGVlUItZZJNsySus7IQvbN4KSvxZhiE"
        "nc5pIcUiTuZZIfcqGafxZS7dxEoWqayUqEtR35TipqzyVA06QuyJei7F2bqeg2AlpxIjE1AqU5mL"
        "XiIXdZSUlQwLiXnVlXgp3DNVy2VgaRyVyRW4WWRJVe6pNV4uRC/lh5Fevi+UXMZVXEuxjJOreCaD"
        "Tmcyl2W1FnGRzMtKgZre1OuhEAWY2MvlNThJyiLNaNNqII6LWlbTGFz2joO+GN3KZMXyOIvr+U28"
        "Fr3PQR+E3Ge4wvaqrMYryArj1nkZp1gzFdhmdi3BgMpqUa5qlaWSt1NXWZyGjp/vBR5XUkJIuRY/"
        "cwZ+iE1sjXfc/H+vnBoZKNGLIRGlWhzFYlZi2SImYV+WqyKNq3XwXnyIiz1wwawd0v+VUHW1SupV"
        "FediWZXgpc6kcoztg9rYDAEbk2yBf4dpvKw1l6DA22HeYRRxzZaRlItlWciiBqnOEaxKkXCxOWg2"
        "lanojeWyFvuv9v+LNHz02hdiJZc5xK/EWZVdZ7mcSZEV4mO8JjkRZ0f7DXfviH0ofJYlgjacQz2Z"
        "agl6WpX/kIW2uFCclGKaSRinmMMUmdgboUUBkVjxVLC30fnxz6Mjmr9gQlYxvMO+KKCfSpDBLSQs"
        "hmV29FY0hoqNVlJBCClbPLhK8hjSg13UTM8TPEReiZ68jZM6YEbw3ugTKl6WbgyklxVxDlXSthVz"
        "ShyXUzgjtBeKCZGGH4skriqo0tAxu8YWmOkleWshSRMpvJLcR2B1UlJfLOdxUWPXvERfXEN5l1kO"
        "1QSGmCp5B9tbzAqINasl7aAqV7O53inYCTtdhJsOSzOKpisypygSGe2OhF+U2qCUGZPCkCAwpbBJ"
        "M8g+6uvd6IGyWC2aESN810/r9TIrZs3zoyyp++JjpvDv6ZKWifNOx7xcxwv80RmfjQ6jn0fn4+PT"
        "E3Eguq/C1+C380Ls/X4fUCMWZWW2+vsS77BwxGETy3qwsD4vGAxYcdCAjX6e9bng52IXdEmh65JY"
        "lTni56WCM4ekQ6J0fDIZnX8YHo5IUsf62ejz6PCnCaQXnQ0nP/wy/JXefdbvhj9Nfjg9P57wsyHJ"
        "VfN6bnzqkP7axS//34ZI64KNZfQmHG1ei6q8UQi9Hzma90x6uFYmXMI9mJAJmYHeHqILDJXNxUXL"
        "Phk3RIjMVms19cWRXEq2ccojqd7sbU0TcjGpVkoHDFi6gq0jGlj2brJaewBHaDZIiBrptayQq2qE"
        "RqIF+ReSfU9sflwuwgZDBGKsT2TidJEpCqma/RhMr6pEdsTjn5g8rQRDFWcVEFwtlzmChDgONSvw"
        "/AIS3PHZzoNtluDutTTsFE8xktl9ATHEYiFp4UzBb6sYmmBRGU6fIkUbET0Yx6Hm/b2vrps5Qj8y"
        "L8tfqqdoOTMIPNl8Dhs11VWZR0hOhYzSrCKdgc8WAIBM+sK9m5bVIhRDnQ4VO9lTPBz3kn4WiH/9"
        "7z/F517WX+ivw96i//dgsKl4lyce/Bi2c+icclBZTLMZYq9NQwu5uCTs9Nna9iOfIVt6XiZx7hvP"
        "e7NljQXkrayQ8J/kDDlrGG56Us8KL9Da++YtSkOwL2Q4C2Fd17AGONxqiQTytIvM4PicCqE+iNlM"
        "piUoAO6wBBKpRKplU9iwBO+dMQUgpqqWMP36SR9RagU1AQPKGwIrQipCHplCnDH20GzxaX+DO9TE"
        "FSDeNf6oPO0Nsc18/Z5Nq1ZPu1uNGmAWtgL04enJyehwYjKni2fm7Q9DvP7Ir7SP2lmT89OP0dnH"
        "4ckoOjo+JxInfzMkdvnarnn468Po/HznRCd+ncpVE6wUbR7SIJzGiahnc8xYs67Em/BtH/+8CzqH"
        "H4fjcTT+6ezs4/FojGXumI9W8gqdDAYbIushQ/ZNybAxR0tmsC3mHjLnA3N2S21g19EEXkCvcAZ2"
        "VV2JsBBl+iRJJ9ABSA6ZjXsIz8KENhqGnet0CkFSWKHC0cvKjfTOT3/5ZsFZJBNavPFtUnQEtsDJ"
        "t4nWEbJIpv9NgtxJ4N7iIMYtj+C176HCOF8r0ihDHMTO2NNIVpBpJ9ICtJPTI8ZmlB2724J5gdiK"
        "gLlmhILiCqUaJ1KeO/51PBl9otkaL3U351rde7O9ejMtF3FWeKSi0w+R/jZmqqXqbpEyGXp35Wrl"
        "dGRqlSH76I9AUw+CRki8BhuyMiU+lbYadvvtBN05oDAQF5S4AHEAi46LJF9RmVqUNx1T8Lwdvnz7"
        "/ct3tudBxSzt05Agi38v3gxt9lMcVsX56NPo6Hg4GbUjpn1MAgEhmWZITl1fxgooESW1BjiASXMZ"
        "5/V8jQy5LuIFCl6Unlz4jQND8ufTH0fR5Pyn8URTvS6vZFRTWu2yyqGttIpvPBDhwQHkJyxmQ+1T"
        "eUBz6WVeGJRq0ISpS1EzVNcyfS6pBZITGKOa9Hg4omBdrxRD0pkcsNI4qkMgz6OYl5hY6VjPMEVg"
        "iw1QGWqhHY9PPxotZKrMfR1oIoqrfK2F/1TWXpUR+Xg0iVperyUPcB+1UlGj9fHk9HwU/Xw8Pv7+"
        "+KOph4yqI1dn/xGF59gVfOXlbySD37v8/Kst0I23/s16snXN4XagCG0QVa5O0q+0HW21vGyUy9IB"
        "FbL8vWljRNYm8fKyLHOIGNBSsjY/xEBTeGBQ3UvxaXyGfxvI2NmxiRMovh1ZhtolubgDYqJiBhBD"
        "wSN7rqUY7IjOyhR8x4XfFdW9UF09mjqyaXYJKkLXYhbroNQELYxY01AmZiIY8LwaeHXV3l+wK9Qc"
        "eAHICOwqq/cmwIGlaGlqOAyLdenGxExRgdKgmaQLNPhgmvPKMYNjCm7soIh5sYW8INagdTMBHiOn"
        "U4oo15LQVhN3yAZ4o4hk4JApGbNsx0lfw85y3DO7YatrreKNUNAomDoZBHMZ3dstLslhCw3It0S0"
        "aUQmtpi5e3W556pXXQP75KxwdpFx/DkSxnoVW1Z5Y6SskxB9i7SUWAKcTOt2zKLMDh25OgSQwWmI"
        "JMi0ONbk8jGuiJbtDVJ6NPEcegNNixayxTImfqZ5iXUPxOvw1a5gzHGUWLiR2WxeE+CuJIRHCanQ"
        "bUkdkf8BNAFLpmbdBbT7BSS520d9yniV1xHVTmW1PiAVBLs8tgknba+lAwzCqkiY7HxbPSVu6+p2"
        "u/hqkkb11aaPpvIzPv3Vfv1KLqJnIbVIavd/te3Prw35iHn7aiKA7lN5XShEjtpi6GCzKdWk1Xm2"
        "HDQtWOozbcVSfgU31GjLDE2zKZ+71DvG27EeRmuJrRGEc7lmk+6JFYWHc81Qf/ODNlpuplKFFudb"
        "/vvCLu1BAzMeJl1D5xQLW22VGYS+HNgW7wUYJPs5IYDqwZU9jSN0UQzjqv6AdHukQRq1KDNoiiow"
        "d2rSbw5XAiQ/7uPTKREfSZjeOherNmHnnA7iS6iPjIZ6sntTPnABWXLNS+nyQEqnEYsM0aTOEgp5"
        "4R+e6GmL46VMtHGSMxfJOqLzO52ToIP9jXBA78D2VGNcH/yZ0KRdig5zfBiZVtm0jrIiIcxceGHn"
        "VfjqOxPAp1kepTcyz9scvOPXi/h24/GrjkHcwAx0lKKSuUxXOecrr5XHNtPXJkZuQzZl2sov8N9d"
        "F3pAqdIdiIssVV/6AJSUm67jHI+u7k2nisdQ6w05H+n3VZ+BPj248khV0Gu5iMArzQUpJWWKr+re"
        "yRDc7emwYGj9j/jp7lUYXt3TOahMjcA0QR6hQOGOEmraF2EY3t+3YzTgQp4lWW2mfaKQnMU566Tk"
        "amwtLldZnkbNYS02SOBgi5NFvGzM2KQ3hjjmYKt9hLtcqTmBCe2M1Kxo5K9C8QtBLDIuQ4fPF6kn"
        "xkeqejFYSF1JfhwrmIEOPFqXAz75ueASkb49nExSvN2ZTJzTOgNv6jWObPEy1qDdt8X/5nEUbW/m"
        "Ml8AkwIGQbgtk30bvjKnEkWCdFG0KGhrreNVFPuP3+gZ2WwRR7ct63/nvUlbb/w51fYaLyCSW0hv"
        "WUL9az7JVfMyZ13VTMvbbwP+qHUwz3KNVlse6ldyyGP6UBHGlcAmCdWExBKWxwvSe1yszXymr9Vn"
        "eWjtw5AHDm6FOLYug5jfIWpUC8Z5aSkJpTBcFb2Nw9YgNMTGstbYB5ZcrTQub4ljK5qKnra+ZtMO"
        "Cb2wgpIp+fV1aS5SULc+q+Dmdo52l8DIcMYpAEiOI8PrfmPtSoxu4VwZRToE0Er+fYVwpDTv2Bbi"
        "YNTir51EDZkP/nb4wJ+dJStEc79jIGSczNs74jxjt8PFjqEH3V4ak3emK24f8Io+qAn6M6HxpnWh"
        "GvHrZH8lKRmQ6A+x8nevgNHhdMiIe3Yidc9FzzpUX8S9OgjEioLT5dpQIwr+AXdoDH+xzGWkBbBD"
        "QubYO4JyyxsqGhl6bneaXPCw7YctSBovLlP4647Zoe34fNkNWsG3izAN+hrXJQoDtnI+jEcVOIOF"
        "UwhEtKxkbo64WlZwwAbtmbxBWq3DevaC9yaSzpqoWvCZEJtrc3zpq/t1uzgjN43rnSHQVgGwF32k"
        "/NJehaH6SPFtEntPgy7DtOuliPbbClbvHHo46w0FhQcVcNvqe8QpoTtbtU5RG6Xc47Q+P4NW08y5"
        "jsEqV6CO2msecUnBko4+Wq/2H4nwmyWSDZYoUZqD2KafZoRHjkwtLQAY0/RojjqNDboF9p0GLmMl"
        "o3pJahF74vXL3us/b00MNkSmgRacRbVA1nc7DsDq+EqSZ26J2SAyp9MNcvsmpifLSCG2AmVt7aLJ"
        "eOZKSmTksr0Ub7TEpmzTmNSZgGZfULpJcuqFtGktt8m0BkBIO0bsAgmX+QZEaF9d2Ceb36vkjLoG"
        "pMTxZKjBDKU8Mcxzc4GKPfeAXmtPG08GOhjyyAMOWiEeI2TXSAv2wWQ4sPWCeSp6nBIVFm8Odt/A"
        "sqb9jeYS5fKg7deWUqs9oF0eq24/dSy23/HLv5prZeumTYcYfUmnHTKfBoSksdzAhtRKYuECrIS/"
        "lVnRS9gpkz5DxEL0et1xl/BtPg0tlxu38R759LqTZjZt5FsmDpuJbq/IP9kUjAXUV+4WJXV5f/9C"
        "clIuzQlGc/Lwh5dzY7OQM2e/DUe3uSK+pEHq1m0o//KUUbNKqmxZe2OMcfldPM60rkv8VK+nue7R"
        "tIeoNfucObZpb+Y1bYjnzGUniWjVB/sKFHn464FBWXsqJsjBGIsLHip/PGoJvCHOiodali+shy71"
        "NSJzU0P7q2blGbciWn087GdVXcbmmh6fpQCvAaL3GSMbho3qHkSU7Qreu0DWghaLBhA3EW7QCpAP"
        "Ct0fFHjRpukuPDixGRBsBaN29fbwyq1hxmgIbw0sLntwbjNAz6LKu+ksvN3vGGBKfibysrxaAef+"
        "vx21CZ9kARw9+/w1Ms7JsdSdVnA7H4GJ4yZHLu06LbNB/CrCLBUHB5bUllmZiOzumFQxlXY/yvWo"
        "qsqqZyYGHcugc/GonD7E6ebhEMcGYsUxa0mHjmJrbzO7t9kOamaDM7PB2SObmz20uVlrYxyH3Z4Q"
        "V1xPkzfFfV23ClbHmNA2MsEF/W07qJ0drDCJkM7vnVhUSyo+yZ2imVVb45sld45vrWxO/0lwisVW"
        "8RWk1lu/Weyb+DVAXKpr3n/PxA0h6UACB26Ku05ogCvnmvOYz2MIN1O1ksuFQmWzWKJiWci4UJqc"
        "PTVkiGfGDRxhyOziS8e7CgjLI8HfsQE9ZG73Hf8ioplyUTRTNrzvi28auSwaC0fx8R8H/EDJ2j0M"
        "2hbSMB3GS0LvvW66opYdnZ3pkj1V3eAbnd87EqBEgOFm59u+srn8lG+XiDva7P1ArIqrgg6sPIp3"
        "Pv37Dd4qy5vLz601aRCXhwT8Kmvy8DtnzttcYlc8yeymkeVgZ7rc3pI9DrpzK97v/eXOrYmtdp+V"
        "e6fdRiRaTsSWLwTilLbvIIbIdMuK0QS1SzffP7qp7c14M+82SN3r3hiT+jaeGC/Y4LIxOgjdiey/"
        "w9w8plXd/edvtR2KvKE7rYNz2WtoF1XYPpf68hxTf5ZdaPj8tHVMu3ebTISIUStsXGX6cJ2OKp9F"
        "6JF9GZoa4t158jAv2nq3erWFqa2wWO9xse5t58+NNUnS7TtxR6Oz0cnR6OTw14c0t+HEKHmGHnS0"
        "w/hyhDlbb5A1/9DlckV/A1ObpxvUGH2/11cX+CcqNXU+0r4+oqAniv0AURSYb65vOVBjWkiCAGHb"
        "jGPvl1YmbzbG8gcUf78OP30Uxy9Pf++Sj3JsRL+c6SU5/cKHzCFoDjd4U3je048b28hUc49Fv+jT"
        "mECDA37Q0XSp2ReRYnp0g8IBvu2qkvuL1LPhkX3RrVBko4Qo6Y7uQXdVT/e+6wbUs556DYL4BhmW"
        "frQTqngqI1quNw183rlCpSP0iI51epgRGN623+gjogcYVBr4+y+cC2TpAeZfdLO0+8U1EvzqmN6H"
        "MyT0rv8Ue/TLZa8J4RXNbqr3EDO7XW+8y60HXhXd+9OfZoHDK5aQGw06F18Cj2kGBwdcT2N2ETjs"
        "YGfzkK2J1jkPLlpu0oSA7YDRxM6D6qLbfIf87K0Cet589+Xq3fXXMexA26+9cAASF137tvtlR3On"
        "HaoMgVa0YiLtYTsp2WsLB1QeA5ywjOxTyInr5WDHVD5APjAz+I/uxqjABUorfitorQI73pOQS5/O"
        "eNwzf5FWF8LsoD3DvNu1j/ZRQ3ty692uyU1Yb9X6MDlLoRmAyXf3/kxudzaFvj+DXmyN9nqm7doe"
        "M+9awvbXtiM1vbZWuhvnRHTQv6VcbVLbpz99EQdbg/WNW1/LmxyYpxsL95/f+dn4XHhXjVt+7G+W"
        "Gx+2reGLmt9syZpaHgfIqs4M+LZCX7zdb4bpvTd5kopT+kWoLfL068xVb4ONcvxnSi+6IO9mBc9r"
        "/2gdkFz8WXTfC9M2bggFW3nBJIJ0tVjqJKWn+yG+L9qJi5DvoLnr1vohqlSuN0CHIVi5vPwtaDUA"
        "vLSJd62r6hs4Aq81MPOne2uFmYrsnxsLeWTurgYNM9c6lFwhw5Od+bRixRmQqIQZXartBfePsc0d"
        "0Z3rXZjFbvVit7QSpnx5jBpft/hG7kk8W5w60XV2Q4qbZ0AKBybILHpmeTKMoC+mBBSrOrqSa6VP"
        "VoLO/wF7Ko0e"
    ),
    "relations": (
        "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"
        "tgnnBw=="
    ),
    "network": (
        "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"
    ),
    "layers": (
        "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"
    ),
    "step": (
        "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"
    ),
    "__init__": (
        "eNptUEFqwzAQvPsVi04JCBPaSyn0EIwphTYpSehVKPa4iMhas5JT2tfXduxTq9NoNDszWqVUUb6d"
        "qGLBIzXCPwgkaCAIFci1nUeLkGxyHKhhoXfbQeh+S6vrJr+jeIFH4rDOlVLZYNBSHjtU4yhLotWx"
        "QrDi+DiQmnZcQ9MBfjLU9MxXSLBDlqaCQ+0mOqO/Z5kpvI1R0yuu8Jq25xt7c/dsaxMnWPdtd4P/"
        "mY0PZixralel9dxb5oi4lF8yT/bsMYsC0hfLZZGce+drM5OaPhEgNmGKXtaR0C1y6YPhAJPE2aH+"
        "eI3zgjTFvm0HFIddVOMvXfNtKvbedhFZsT+U5qM8HF/2O3oitckfVPYLdfqVVA=="
    ),
}


def _load_embedded_core():
    """Decode, hash-check and execute the embedded cemt_core modules.

    Returns (package_module, hash_report). Modules are registered in
    sys.modules under cemt_core.* so their relative imports resolve to each
    other and to nothing on disk.
    """
    import types
    import zlib as _z
    import base64 as _b
    report = {}
    pkg = types.ModuleType("cemt_core")
    pkg.__path__ = []
    pkg.__package__ = "cemt_core"
    sys.modules["cemt_core"] = pkg
    for name in ("spec", "relations", "network", "layers", "step", "__init__"):
        src = _z.decompress(_b.b64decode(EMBEDDED_SOURCES[name]))
        report[name] = hashlib.sha256(src).hexdigest() == EMBEDDED_HASHES[name]
        code = compile(src, f"<embedded cemt_core/{name}.py>", "exec")
        if name == "__init__":
            exec(code, pkg.__dict__)
            continue
        mod = types.ModuleType(f"cemt_core.{name}")
        mod.__package__ = "cemt_core"
        sys.modules[f"cemt_core.{name}"] = mod
        setattr(pkg, name, mod)
        exec(code, mod.__dict__)
    return pkg, report



# ############################################################################
# PART 1: DRIVER
# ############################################################################

_CORE = None


def _core():
    """Load the embedded frozen cemt_core once per process."""
    global _CORE
    if _CORE is None:
        pkg, report = _load_embedded_core()
        sp = sys.modules["cemt_core.spec"]
        _CORE = SimpleNamespace(
            generate_spec=pkg.generate_spec, build_network=pkg.build_network,
            run_one_trial=pkg.run_one_trial, CORE_VERSION=pkg.CORE_VERSION,
            AblationSpec=sp.AblationSpec, AdaptationSpec=sp.AdaptationSpec,
            TimeSpec=sp.TimeSpec, RelationClass=sp.RelationClass,
            Condition=sp.Condition, Node=sp.Node, Relation=sp.Relation,
            Governance=sp.Governance, CLASS_SUPPLIES=sp.CLASS_SUPPLIES,
            hash_report=report)
    return _CORE


def freeze_status() -> dict:
    c = _core()
    ok = all(c.hash_report.values())
    bad = [k for k, v in c.hash_report.items() if not v]
    return dict(core_version=c.CORE_VERSION,
                freeze_digest=EMBEDDED_SET_DIGEST if ok else None, verified=ok,
                note=("embedded core matches freeze record" if ok
                      else "EMBEDDED CORE DIFFERS from freeze record: " + ", ".join(bad)))


def _draw(rng, lo, hi, scale):
    if scale == "lin":
        return float(rng.uniform(lo, hi))
    if scale == "log":
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    if scale == "int":
        return int(rng.integers(int(lo), int(hi) + 1))
    return int(round(np.exp(rng.uniform(np.log(lo), np.log(hi)))))


def sample_params(rng, regime: str) -> dict:
    p = {name: _draw(rng, lo, hi, sc) for name, lo, hi, sc, _ in SAMPLED}
    for name, lo, hi, sc, _ in OPEN_SAMPLED:
        v = _draw(rng, lo, hi, sc)        # always drawn so the stream is aligned
        p[name] = v if regime == "open" else FIXED["closed_phantom_rates"]
    return p


def oat_levels(name: str, n: int) -> List[float]:
    lo, hi, scale = next((s[1], s[2], s[3]) for s in SAMPLED if s[0] == name)
    v = (np.exp(np.linspace(np.log(lo), np.log(hi), n)) if scale in ("log", "logint")
         else np.linspace(lo, hi, n))
    if scale in ("int", "logint"):
        return sorted(set(int(round(x)) for x in v))
    return [float(x) for x in v]


def apply_conditions(spec, params: dict, seed: int) -> None:
    """Driver scenario conditions (decision 1). Mechanics untouched.

    Per estate node, I, X and A presence are drawn from an equicorrelated
    Gaussian copula: z_k = sqrt(rho) c + sqrt(1 - rho) e_k, condition present
    iff z_k < Phi^-1(rate_k). Marginal rates are preserved; rho sets how far
    the three conditions co-locate on the same nodes. Conferring control
    planes are then added (Paper 1B Authority row, conferring form).
    """
    c = _core()
    rng = np.random.default_rng(seed * 7 + 13)
    est = [n for n in spec.nodes if n.governance == "estate"]
    rho = float(params["colocation"])
    common = rng.standard_normal(len(est))
    rates = (params["interface_rate"], params["x_presence_rate"], params["a_presence_rate"])
    pres = []
    for r in rates:
        z = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * rng.standard_normal(len(est))
        pres.append(z < norm.ppf(r))
    for k, n in enumerate(est):
        n.interface = bool(pres[0][k])
        n.execution_pathway = bool(pres[1][k])
        n.authority = bool(pres[2][k])
        zs = [z for z in n.zones if z != 0]
        n.zones = ([0] + zs) if n.interface else zs
    ids = [n.id for n in est]
    for k in range(int(params["n_conferring_planes"])):
        if rng.random() < params["frac_external_issuers"]:
            issuer = f"idp{k}"
            spec.nodes.append(c.Node(id=issuer, governance="vendor", interface=True,
                                     visible=False, state_object="credentials"))
        else:
            issuer = ids[int(rng.integers(len(ids)))]
        m = max(1, int(round(params["conferring_span"] * len(ids))))
        for mem in rng.choice(ids, min(m, len(ids)), replace=False):
            if mem != issuer:
                spec.relations.append(c.Relation(
                    issuer, str(mem), c.Condition.AUTHORITY,
                    c.RelationClass.CONTROL_PLANE_CONFERRING,
                    conferral=False, group=f"conf{k}"))


def build_spec(params: dict, scenario_seed: int):
    c = _core()
    target = {s[0]: s[4] for s in SAMPLED + OPEN_SAMPLED}
    gen = {k: v for k, v in params.items() if target.get(k) in ("gen", "rate")}
    gen.update(node_count=FIXED["node_count"], n_zones=FIXED["n_zones"],
               phantom_dep_rate=0.0)
    spec = c.generate_spec(gen, seed=scenario_seed, scenario_id=f"p4a_{scenario_seed}")
    apply_conditions(spec, params, scenario_seed)
    sp = sys.modules["cemt_core.spec"]
    actions = [sp.DefenderActionKind.REMEDIATE]
    if int(params["revoke_enabled"]):
        actions.append(sp.DefenderActionKind.REVOKE_TRUST)
    spec.adaptation = c.AdaptationSpec(
        remediation_capability=float(params["remediation_capability"]),
        synchrony=FIXED["synchrony"], threshold=float(params["policy_threshold"]),
        tau_a=FIXED["tau_a"], simple_policy=FIXED["simple_policy"],
        actions_allowed=actions)
    spec.entry_certain = FIXED["entry_certain"]
    spec.time = c.TimeSpec(latency_steps=int(params["latency_steps"]),
                           drift_increment=float(params["drift_increment"]),
                           exfil_dwell_steps=FIXED["exfil_dwell_steps"],
                           max_steps=FIXED["max_steps"])
    spec.ablation = c.AblationSpec(structure=True, time=True, adaptation=True)
    return spec


def measure_structure(spec) -> dict:
    """Realised measurements from the scenario specification (decision D3:
    derived from the relation table, never parameters). Definitions follow
    cemt_core.relations: one supply row per condition a relation supplies,
    connection rows symmetric, node-level instances as self-supply."""
    c = _core()
    RC, CS = c.RelationClass, c.CLASS_SUPPLIES
    gov_ctl = {g.id: g.defender_controlled for g in spec.governance}
    gov = {n.id: n.governance for n in spec.nodes}
    est = [n for n in spec.nodes if gov_ctl[n.governance]]
    est_ids = {n.id for n in est}
    n_est = max(len(est), 1)

    supply = {}                       # (receiver, cond) -> {supplier: class}
    fan = {}                          # (supplier, class) -> set(receivers)
    fan_sos = {}
    n_rows = n_sos = 0

    def _row(s, r, cond, rc):
        nonlocal n_rows, n_sos
        supply.setdefault((r, cond), {}).setdefault(s, rc)
        if s != r:
            n_rows += 1
            fan.setdefault((s, rc), set()).add(r)
            if gov[s] != gov[r]:
                n_sos += 1
                fan_sos.setdefault(s, set()).add(r)

    counts = {rc: 0 for rc in RC}
    conn_pairs = set()
    for rel in spec.relations:
        counts[rel.relation_class] += 1
        for cond in CS[rel.relation_class]:
            _row(rel.supplier, rel.receiver, cond, rel.relation_class)
            if rel.relation_class == RC.CONNECTION:
                _row(rel.receiver, rel.supplier, cond, rel.relation_class)
        if rel.relation_class == RC.CONNECTION and rel.supplier in est_ids and rel.receiver in est_ids:
            conn_pairs.add(tuple(sorted((rel.supplier, rel.receiver))))
    for n in spec.nodes:
        for cond, present in (("I", n.interface), ("X", n.execution_pathway), ("A", n.authority)):
            if present:
                supply.setdefault((n.id, cond), {}).setdefault(n.id, RC.CONNECTION)

    cut_cond = {"I": 0, "X": 0, "A": 0}
    cut_cls = {rc: 0 for rc in RC}
    cut_sos = 0
    cut_by_supplier = {}
    cap = 0
    for n in est:
        ok = True
        for cond in ("I", "X", "A"):
            s = supply.get((n.id, cond), {})
            if not s:
                ok = False
            if len(s) == 1:
                sup, rc = next(iter(s.items()))
                if sup != n.id:
                    cut_cond[cond] += 1
                    cut_cls[rc] += 1
                    cut_by_supplier[sup] = cut_by_supplier.get(sup, 0) + 1
                    if gov[sup] != gov[n.id]:
                        cut_sos += 1
        cap += ok

    def _fan(rc):
        v = [len(r) for (s, k), r in fan.items() if k == rc]
        return v or [0]

    tot = {}
    for (s, _), r in fan.items():
        tot.setdefault(s, set()).update(r)
    return dict(
        n_estate_nodes=len(est), n_external_nodes=len(spec.nodes) - len(est),
        I_real=float(np.mean([n.interface for n in est])),
        X_real=float(np.mean([n.execution_pathway for n in est])),
        A_real=float(np.mean([n.authority for n in est])),
        vis_real=float(np.mean([n.visible for n in est])),
        C_IAX=float(np.mean([n.interface and n.execution_pathway and n.authority for n in est])),
        capability_supplied=cap / n_est,
        conn_density_realised=len(conn_pairs) / max(n_est * (n_est - 1) / 2, 1),
        channel_per_node=counts[RC.CHANNEL] / n_est,
        directing_per_node=counts[RC.CONTROL_PLANE_DIRECTING] / n_est,
        conferring_per_node=counts[RC.CONTROL_PLANE_CONFERRING] / n_est,
        fanout_max_total=max((len(r) for r in tot.values()), default=0),
        fanout_mean_directing=float(np.mean(_fan(RC.CONTROL_PLANE_DIRECTING))),
        fanout_max_directing=max(_fan(RC.CONTROL_PLANE_DIRECTING)),
        fanout_max_conferring=max(_fan(RC.CONTROL_PLANE_CONFERRING)),
        fanout_max_sos=max((len(r) for r in fan_sos.values()), default=0),
        cut_I_per_node=cut_cond["I"] / n_est, cut_X_per_node=cut_cond["X"] / n_est,
        cut_A_per_node=cut_cond["A"] / n_est,
        cut_connection_per_node=cut_cls[RC.CONNECTION] / n_est,
        cut_channel_per_node=cut_cls[RC.CHANNEL] / n_est,
        cut_directing_per_node=cut_cls[RC.CONTROL_PLANE_DIRECTING] / n_est,
        cut_conferring_per_node=cut_cls[RC.CONTROL_PLANE_CONFERRING] / n_est,
        cut_sos_per_node=cut_sos / n_est,
        cut_max_single_supplier=max(cut_by_supplier.values(), default=0),
        sos_row_fraction=n_sos / max(n_rows, 1),
    )


def evaluate_scenario(task: dict) -> dict:
    """One scenario: build once, measure, run n_trials. Runs in a worker."""
    c = _core()
    seed = task["scenario_seed"]
    spec = build_spec(task["params"], seed)
    meas = measure_structure(spec)
    net = c.build_network(spec, np.random.default_rng(seed))
    col, xmax, ext, steps, entry = [], [], [], [], []
    for t in range(task["n_trials"]):
        r = c.run_one_trial(net, np.random.default_rng(seed * 100_003 + t))
        e = any(v == "entry" for _, _, v in r["reached"])
        entry.append(e)
        col.append(r["collapsed"])
        xmax.append(r["x_true_max"])
        ext.append(r["ever_reached_fraction"])
        steps.append(r["n_steps"])
    row = dict(task["meta"])
    row.update(task["params"])
    row.update(meas)
    row.update(tau_a=FIXED["tau_a"], node_count=FIXED["node_count"],
               n_trials=task["n_trials"], n_collapsed=int(np.sum(col)),
               trial_collapse_rate=float(np.mean(col)),
               n_entry_compromised=int(np.sum(entry)),
               entry_compromise_rate=float(np.mean(entry)),
               conditional_collapse_rate=(float(np.sum(np.array(col) & np.array(entry)) / np.sum(entry))
                                          if np.sum(entry) >= FIXED["conditional_min_entries"] else float("nan")),
               mean_x_true_max=float(np.mean(xmax)),
               mean_ever_reached=float(np.mean(ext)),
               mean_steps=float(np.mean(steps)), core_version=c.CORE_VERSION)
    return row


def _run_tasks(tasks: List[dict], path: str, workers: int, label: str) -> None:
    """Run in parallel, append each finished row to `path`; resumable."""
    done = set(pd.read_csv(path)["task_key"].astype(str)) if os.path.exists(path) else set()
    todo = [t for t in tasks if t["meta"]["task_key"] not in done]
    print(f"  {label}: {len(tasks)} tasks, {len(done)} done, {len(todo)} to run")
    if not todo:
        return
    t0, n = time.time(), 0

    def _write(row):
        pd.DataFrame([row]).to_csv(path, mode="a", index=False, header=not os.path.exists(path))

    def _progress():
        el = time.time() - t0
        if n in (1, 5, 20) or n % max(1, len(todo) // 20) == 0 or n == len(todo):
            print(f"    {n}/{len(todo)}  elapsed {el/60:.1f} min  "
                  f"eta {el/n*(len(todo)-n)/60:.1f} min", flush=True)

    if workers <= 1:
        for t in todo:
            _write(evaluate_scenario(t)); n += 1; _progress()
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for f in as_completed([ex.submit(evaluate_scenario, t) for t in todo]):
                _write(f.result()); n += 1; _progress()


def sample_plan(design: dict) -> List[Tuple[str, int, str]]:
    """(sample role, seed, regime) for every validation file."""
    plan = [("development", design["dev_seed"], "closed")]
    plan += [("replication", s, "closed") for s in design["replication_seeds"]]
    plan += [("heldout_regime", design["open_seed"], "open")]
    return plan


def generate(design: dict, run_dir: str, workers: int) -> None:
    for role, seed, regime in sample_plan(design):
        rng = np.random.default_rng(seed)
        tasks = []
        for i in range(design["n_scenarios"]):
            params = sample_params(rng, regime)
            sid = seed * 10000 + i + 1
            tasks.append(dict(params=params, scenario_seed=sid, n_trials=design["n_trials"],
                              meta=dict(task_key=str(sid), scenario_id=sid, seed=seed,
                                        role=role, regime=regime)))
        _run_tasks(tasks, os.path.join(run_dir, f"validation_{role}_{regime}_seed{seed}.csv"),
                   workers, f"{role} ({regime}) seed {seed}")
    if design["oat"]:
        tasks = []
        for name, *_ in SAMPLED:
            for lv in oat_levels(name, design["oat_levels"]):
                params = dict(BASELINE, **{name: lv},
                              phantom_cp_rate=0.0, phantom_channel_rate=0.0)
                tasks.append(dict(params=params, scenario_seed=OAT_SEED,
                                  n_trials=design["oat_trials"],
                                  meta=dict(task_key=f"{name}={lv}", param_name=name, level=lv)))
        _run_tasks(tasks, os.path.join(run_dir, "oat_sweep.csv"), workers, "OAT sweep")


# ---- pre-specification: written before scoring, enforced at analysis -------

def code_hash() -> str:
    """SHA-256 of this file with the RUN_CONFIG block removed and line endings
    normalised, so changing mode or paths does not change the hash but any
    change to measurement, model, analysis or decision rules does."""
    src = open(os.path.abspath(__file__), "rb").read().replace(b"\r\n", b"\n").decode("utf-8")
    src = re.sub(r"# >>> RUN_CONFIG.*?# <<< RUN_CONFIG", "", src, flags=re.S)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def prespecification(design: dict, fz: dict) -> dict:
    return json.loads(json.dumps(dict(
        code_version=CODE_VERSION, code_sha256=code_hash(),
        core_version=fz["core_version"], freeze_digest=fz["freeze_digest"],
        design=design, sampled=SAMPLED, open_sampled=OPEN_SAMPLED, fixed=FIXED,
        baseline=BASELINE, feature_sets=FEATURE_SETS, estimator=ESTIMATOR,
        formulas={k: v["expr"] for k, v in FORMULAS.items()}, additive=ADD_EXPR,
        column_map=COLUMN_MAP, h1_axes=H1_AXES,
        outcome=("scenario collapse rate over all trials; trial collapse = x_true >= 0.30 sustained "
                 "3 steps; penetration assumed at a uniform Interface node, compromise gated"),
        measure="Harrell concordance C, paired bootstrap over scenarios, two-sided",
        decision=("pass iff Delta C >= min_delta_c and BH-adjusted p < alpha; BH family = "
                  "all confirmatory comparisons H1-H4; H1 and H4 pass only if all parts pass"),
    ), default=str))


def check_prespecification(run_dir: str, current: dict) -> Tuple[bool, List[str]]:
    path = os.path.join(run_dir, "P4A_prespecification.json")
    if not os.path.exists(path):
        return False, ["no pre-specification record in the run directory"]
    saved = json.load(open(path))
    diffs = [k for k in set(saved) | set(current) if saved.get(k) != current.get(k)]
    return not diffs, diffs


def write_prespecification(run_dir: str, current: dict) -> None:
    path = os.path.join(run_dir, "P4A_prespecification.json")
    if os.path.exists(path):
        ok, diffs = check_prespecification(run_dir, current)
        if not ok:
            raise SystemExit(f"{path} differs from the current file in: {', '.join(sorted(diffs))}. "
                             "Start a new run_id, or restore the version it was written with.")
        return
    with open(path, "w") as f:
        json.dump(current, f, indent=2)


# ############################################################################
# PART 2: ANALYSIS
# ############################################################################

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_samples(run_dir: str) -> pd.DataFrame:
    files = sorted(f for f in os.listdir(run_dir) if f.startswith("validation_") and f.endswith(".csv"))
    if not files:
        raise SystemExit(f"No validation_*.csv files in {run_dir}.")
    df = pd.concat([pd.read_csv(os.path.join(run_dir, f)).assign(source_file=f) for f in files],
                   ignore_index=True)
    for canon, col in COLUMN_MAP.items():
        if col in df.columns:
            df[canon] = df[col]
    return df


# ---- scoring ---------------------------------------------------------------

def add_reference(dev: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """ADD standardisation constants, fixed once on the development sample."""
    return {a: (float(dev[a].mean()), float(dev[a].std(ddof=0))) for a in ADD_POS + ADD_NEG}


def add_score(df: pd.DataFrame, ref) -> np.ndarray:
    tot = np.zeros(len(df))
    for sign, axes in ((1, ADD_POS), (-1, ADD_NEG)):
        for a in axes:
            mu, sd = ref[a]
            if sd > 1e-12:
                tot += sign * (df[a].values - mu) / sd
    return tot


def formula_score(df: pd.DataFrame, name: str) -> np.ndarray:
    ns = SimpleNamespace(**{v: df[v].values.astype(float) for v in FORMULAS[name]["vars"]})
    return np.asarray(FORMULAS[name]["f"](ns), float)


def formula_status(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, spec in FORMULAS.items():
        absent = [v for v in spec["vars"] if v not in df.columns]
        const = [v for v in spec["vars"] if v in df.columns and float(df[v].std()) <= 1e-12]
        rows.append(dict(formula=name, role=spec["role"], expr=spec["expr"],
                         status="NOT EVALUABLE" if absent else "DEGENERATE" if const else "OK",
                         absent_inputs=";".join(absent), constant_inputs=";".join(const)))
    rows.append(dict(formula="ADD", role="additive", expr=ADD_EXPR, status="OK",
                     absent_inputs="", constant_inputs=""))
    return pd.DataFrame(rows)


def _logit_rate(df: pd.DataFrame) -> np.ndarray:
    p = (df["n_collapsed"].values + 0.5) / (df["n_trials"].values + 1.0)
    return np.log(p / (1 - p))


def fit_models(dev: pd.DataFrame, seed: int) -> Dict[str, object]:
    """Every fitted model is trained once on the development sample."""
    y = _logit_rate(dev)
    models = {}
    for name, cols in FEATURE_SETS.items():
        m = HistGradientBoostingRegressor(
            max_iter=ESTIMATOR["max_iter"], learning_rate=ESTIMATOR["learning_rate"],
            max_leaf_nodes=ESTIMATOR["max_leaf_nodes"], min_samples_leaf=ESTIMATOR["min_samples_leaf"],
            l2_regularization=ESTIMATOR["l2_regularization"], random_state=seed)
        models[name] = (m.fit(dev[cols].values, y), cols, False)
    six = H1_AXES
    models["fit_six_linear"] = (make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(dev[six].values, y), six, False)
    models["fit_six_log"] = (make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
        np.log(np.maximum(dev[six].values, 1e-6)), y), six, True)
    return models


def model_score(models, name, df) -> np.ndarray:
    m, cols, log = models[name]
    X = df[cols].values
    return m.predict(np.log(np.maximum(X, 1e-6)) if log else X)


def all_scores(df, names, add_ref, models) -> Dict[str, np.ndarray]:
    out = {}
    for n in names:
        out[n] = add_score(df, add_ref) if n == "ADD" else formula_score(df, n)
    for n in models:
        out[n] = model_score(models, n, df)
    return out


# ---- concordance and paired bootstrap --------------------------------------

def concordance(s: np.ndarray, y: np.ndarray) -> float:
    """Harrell's C between a score and a continuous outcome: probability that
    the pair with the higher outcome has the higher score, over pairs with
    different outcomes; score ties count one half."""
    sy = np.sign(y[:, None] - y[None, :])
    denom = np.abs(sy).sum()
    if denom == 0:
        return float("nan")
    return float(0.5 + 0.5 * (np.sign(s[:, None] - s[None, :]) * sy).sum() / denom)


def bootstrap_c(scores: Dict[str, np.ndarray], y: np.ndarray, n_boot: int, seed: int):
    """Point C and bootstrap C per score, all scores on the same resamples."""
    y = y.astype(np.float32)
    names = list(scores)
    S = {k: np.asarray(v, np.float32) for k, v in scores.items()}
    point = {k: concordance(S[k], y) for k in names}
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, len(names)))
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        sy = np.sign(yb[:, None] - yb[None, :])
        denom = np.abs(sy).sum()
        for j, k in enumerate(names):
            sb = S[k][idx]
            boots[b, j] = 0.5 + 0.5 * (np.sign(sb[:, None] - sb[None, :]) * sy).sum() / denom
    return point, pd.DataFrame(boots, columns=names)


def paired(point, boots, a, b) -> dict:
    d = boots[a] - boots[b]
    diff = point[a] - point[b]
    p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    p = max(p, 1.0 / (len(d) + 1))
    return dict(c_a=point[a], c_b=point[b], delta_c=diff,
                ci_lo=float(np.percentile(d, 2.5)), ci_hi=float(np.percentile(d, 97.5)), p=p)


def bh_adjust(p: List[float]) -> List[float]:
    p = np.asarray(p, float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        prev = min(prev, p[idx] * m / (m - rank))
        adj[idx] = prev
    return adj.tolist()


# ---- confirmatory ----------------------------------------------------------

CONFIRMATORY = (
    [("H1", "ADD", s, "replication") for s in ["S_I", "S_A", "S_X", "S_T", "S_rem", "S_vis"]]
    + [("H2", "F1", "ADD", "replication"),
       ("H2c", "FC", "F1", "replication"),
       ("H3", "Phi_op", "Phi_conj", "replication")]
    + [("H4", "Phi_STA", b, "heldout_regime") for b in ["Phi_ST", "Phi_SA", "Phi_TA"]]
)


def confirmatory(evals: Dict[str, tuple], design: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for hyp, a, b, sample in CONFIRMATORY:
        point, boots = evals[sample]
        rows.append(dict(hypothesis=hyp, comparison=f"{a} > {b}", sample=sample,
                         **paired(point, boots, a, b)))
    t = pd.DataFrame(rows)
    t["p_bh"] = bh_adjust(t["p"].tolist())
    t["pass"] = (t["delta_c"] >= design["min_delta_c"]) & (t["p_bh"] < design["alpha"])
    verdict = (t.groupby("hypothesis")
               .agg(n_comparisons=("pass", "size"), n_pass=("pass", "sum"))
               .reset_index())
    verdict["verdict"] = np.where(verdict["n_pass"] == verdict["n_comparisons"], "SUPPORTED", "NOT SUPPORTED")
    return t, verdict


# ---- secondary: operational 0.15 label, AUC and DeLong ----------------------

def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N); out[J] = T
    return out


def delong_test(y, a, b) -> dict:
    y = np.asarray(y).astype(int)
    order = np.argsort(-y)
    ys = y[order]
    P = np.vstack([np.asarray(a, float)[order], np.asarray(b, float)[order]])
    m = int(ys.sum()); n = len(ys) - m
    if m == 0 or n == 0:
        return dict(auc_a=np.nan, auc_b=np.nan, diff=np.nan, z=np.nan, p=np.nan)
    tx = np.array([_midrank(r[:m]) for r in P]); ty = np.array([_midrank(r[m:]) for r in P])
    tz = np.array([_midrank(r) for r in P])
    aucs = tz[:, :m].sum(1) / (m * n) - (m + 1.0) / (2.0 * n)
    cov = np.atleast_2d(np.cov((tz[:, :m] - tx) / n) / m + np.cov(1.0 - (tz[:, m:] - ty) / m) / n)
    se = float(np.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 0.0)))
    z = (aucs[0] - aucs[1]) / se if se > 0 else 0.0
    return dict(auc_a=float(aucs[0]), auc_b=float(aucs[1]), diff=float(aucs[0] - aucs[1]),
                z=float(z), p=float(2 * (1 - norm.cdf(abs(z)))))


def secondary_auc(df, scores, design) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Operational outcome definition: scenario positive iff collapse rate >=
    threshold. Fixed-form scores only, development sample."""
    rows, dl = [], []
    for thr in design["secondary_thresholds"]:
        y = (df["trial_collapse_rate"].values >= thr).astype(int)
        npos, nneg = int(y.sum()), int((1 - y).sum())
        if min(npos, nneg) < 3:
            continue
        for k, s in scores.items():
            nv = len(FORMULAS[k]["vars"]) if k in FORMULAS else 6
            epv = min(npos, nneg) / nv
            rows.append(dict(threshold=thr, score=k, auc=roc_auc_score(y, s), n_pos=npos,
                             n_neg=nneg, epv_min_class=epv))
        for a, b in [("ADD", s) for s in ["S_I", "S_A", "S_X", "S_T", "S_rem", "S_vis"]] + \
                    [("F1", "ADD"), ("FC", "F1")]:
            dl.append(dict(threshold=thr, pair=f"{a} vs {b}", **delong_test(y, scores[a], scores[b])))
    dl = pd.DataFrame(dl)
    if len(dl):
        dl["p_bh"] = bh_adjust(dl["p"].tolist())
    return pd.DataFrame(rows), dl


def interaction_screen(df, axes, thr, n_perm, seed) -> pd.DataFrame:
    """Additive vs interaction logistic per pair, standardisation inside each
    fold, empirical permutation p-value. Secondary, operational label."""
    y = (df["trial_collapse_rate"].values >= thr).astype(int)
    if min(int(y.sum()), int((1 - y).sum())) < 10:
        return pd.DataFrame()
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    add_m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    int_m = make_pipeline(StandardScaler(), PolynomialFeatures(2, interaction_only=True, include_bias=False),
                          LogisticRegression(max_iter=500))

    def lift(X, yy):
        pa = cross_val_predict(add_m, X, yy, cv=cv, method="predict_proba")[:, 1]
        pi = cross_val_predict(int_m, X, yy, cv=cv, method="predict_proba")[:, 1]
        return roc_auc_score(yy, pi) - roc_auc_score(yy, pa)

    rng = np.random.default_rng(seed)
    axes = [a for a in axes if a in df.columns and df[a].std() > 1e-12]
    rows = []
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            X = df[[axes[i], axes[j]]].values
            obs = lift(X, y)
            null = np.array([lift(X, rng.permutation(y)) for _ in range(n_perm)])
            rows.append(dict(axis_i=axes[i], axis_j=axes[j], interaction_lift=obs,
                             null_mean=float(null.mean()),
                             p_perm=float((1 + (null >= obs).sum()) / (1 + n_perm))))
    out = pd.DataFrame(rows)
    out["p_bh"] = bh_adjust(out["p_perm"].tolist())
    return out.sort_values("p_perm")


def oat_profile_similarity(sw: pd.DataFrame, n_grid=20) -> pd.DataFrame:
    """Exploratory: correlation between normalised one-at-a-time response
    profiles of collapse rate. Not a global identifiability analysis."""
    prof = {}
    for param, g in sw.groupby("param_name"):
        v = g.sort_values("level")["trial_collapse_rate"].values.astype(float)
        if len(v) >= 2 and np.ptp(v) > 1e-9:
            v = (v - v.min()) / np.ptp(v)
            prof[param] = np.interp(np.linspace(0, 1, n_grid), np.linspace(0, 1, len(v)), v)
    names = sorted(prof)
    if len(names) < 2:
        return pd.DataFrame()
    M = np.corrcoef(np.vstack([prof[n] for n in names]))
    return pd.DataFrame([dict(param_i=names[i], param_j=names[j], profile_corr=float(M[i, j]))
                         for i in range(len(names)) for j in range(i + 1, len(names))]
                        ).sort_values("profile_corr", key=np.abs, ascending=False)


# ---- figures and report ----------------------------------------------------

def plot_ladder(ctab: pd.DataFrame, path: str) -> None:
    order = ["best_single_descriptive", "ADD", "F1", "FC", "Phi_conj", "Phi_rel", "Phi_op"]
    t = ctab[ctab.score.isin(order)].set_index("score").reindex(order).dropna(subset=["c"])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(t))
    ax.errorbar(x, t["c"], yerr=[t["c"] - t["ci_lo"], t["ci_hi"] - t["c"]], fmt="o", color="#1D4ED8")
    ax.set_xticks(x)
    ax.set_xticklabels(t.index, rotation=20)
    ax.set_ylabel("concordance C with collapse rate")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_roc(df, scores, names, thr, path) -> None:
    y = (df["trial_collapse_rate"].values >= thr).astype(int)
    if min(int(y.sum()), int((1 - y).sum())) < 3:
        return
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    for n in names:
        fpr, tpr, _ = roc_curve(y, scores[n])
        ax.plot(fpr, tpr, lw=1.6, label=f"{n} (AUC {roc_auc_score(y, scores[n]):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#999999")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def write_summary(path, verdict, conf, ctab, status, fz, pre_ok, pre_diffs, design) -> None:
    L = [f"# Paper 4A analysis, {CODE_VERSION}", ""]
    L.append(f"- Substrate: embedded cemt_core {fz['core_version']}, {fz['note']}")
    L.append(f"- Pre-specification: {'matches' if pre_ok else 'DOES NOT MATCH (' + ', '.join(pre_diffs) + ')'}")
    L.append(f"- Decision rule: Delta C >= {design['min_delta_c']} and BH-adjusted p < "
             f"{design['alpha']} (BH over all {len(conf)} confirmatory comparisons)")
    L += ["", "## Verdicts", "", "| Hypothesis | Comparisons passing | Verdict |", "|---|---|---|"]
    for _, r in verdict.iterrows():
        L.append(f"| {r.hypothesis} | {r.n_pass}/{r.n_comparisons} | {r.verdict} |")
    L += ["", "## Confirmatory comparisons", "",
          "| H | Comparison | Sample | C (a) | C (b) | Delta C | 95% CI | p | p (BH) | Pass |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in conf.iterrows():
        L.append(f"| {r.hypothesis} | {r.comparison} | {r['sample']} | {r.c_a:.3f} | {r.c_b:.3f} | "
                 f"{r.delta_c:+.3f} | [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | {r.p:.3g} | {r.p_bh:.3g} | {r['pass']} |")
    L += ["", "## Concordance by score and sample (descriptive)", "",
          "| Sample | Score | C | 95% CI |", "|---|---|---|---|"]
    for _, r in ctab.iterrows():
        L.append(f"| {r['sample']} | {r.score} | {r.c:.3f} | [{r.ci_lo:.3f}, {r.ci_hi:.3f}] |")
    L += ["", "## Candidate status", "", "| Formula | Role | Status | Constant inputs |", "|---|---|---|---|"]
    for _, r in status.iterrows():
        L.append(f"| {r.formula} | {r.role} | {r.status} | {r.constant_inputs or '-'} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def analyse(run_dir: str, design: dict, fz: dict, pre_ok: bool, pre_diffs: List[str]) -> dict:
    p = lambda name: os.path.join(run_dir, f"P4A_{name}")
    df = load_samples(run_dir)
    dev = df[df.role == "development"].reset_index(drop=True)
    rep = df[df.role == "replication"].reset_index(drop=True)
    opn = df[df.role == "heldout_regime"].reset_index(drop=True)
    if dev.empty or rep.empty or opn.empty:
        raise SystemExit("Development, replication and held-out regime samples are all required.")

    status = formula_status(dev)
    status.to_csv(p("candidate_status.csv"), index=False)
    fixed_names = [r.formula for r in status.itertuples() if r.status == "OK"]
    print(f"\n{CODE_VERSION}: dev {len(dev)}, replication {len(rep)}, held-out regime {len(opn)}; "
          f"fixed forms evaluable: {len(fixed_names)}")

    add_ref = add_reference(dev)
    pd.DataFrame([dict(axis=k, mean=v[0], sd=v[1]) for k, v in add_ref.items()]).to_csv(
        p("ADD_standardisation.csv"), index=False)
    print("  fitting models on the development sample...")
    models = fit_models(dev, design["model_seed"])

    evals, ctab_rows = {}, []
    for label, d in (("development", dev), ("replication", rep), ("heldout_regime", opn)):
        sc = all_scores(d, fixed_names, add_ref, models)
        six = {k: sc[k] for k in ["S_I", "S_A", "S_X", "S_T", "S_rem", "S_vis"]}
        best = max(six, key=lambda k: concordance(six[k], d["trial_collapse_rate"].values))
        sc["best_single_descriptive"] = six[best]
        print(f"  bootstrap concordance on {label} ({design['n_boot']} resamples)...")
        point, boots = bootstrap_c(sc, d["trial_collapse_rate"].values, design["n_boot"], design["boot_seed"])
        evals[label] = (point, boots)
        for k in sc:
            ctab_rows.append(dict(sample=label, score=k, c=point[k],
                                  ci_lo=float(np.percentile(boots[k], 2.5)),
                                  ci_hi=float(np.percentile(boots[k], 97.5)),
                                  note=(f"= {best}" if k == "best_single_descriptive" else "")))
    ctab = pd.DataFrame(ctab_rows)
    ctab.to_csv(p("concordance_by_score.csv"), index=False)

    conf, verdict = confirmatory(evals, design)
    conf.to_csv(p("confirmatory.csv"), index=False)
    verdict.to_csv(p("verdicts.csv"), index=False)

    rep_rows = []
    for seed, d in rep.groupby("seed"):
        d = d.reset_index(drop=True)
        sc = all_scores(d, ["ADD", "F1", "FC"] + ["S_I", "S_A", "S_X", "S_T", "S_rem", "S_vis"], add_ref, {})
        y = d["trial_collapse_rate"].values
        rep_rows.append(dict(seed=seed, **{f"C_{k}": concordance(v, y) for k, v in sc.items()}))
    pd.DataFrame(rep_rows).to_csv(p("replication_by_seed.csv"), index=False)

    # Descriptive decomposition from the same trials: entry compromise and
    # collapse given entry compromise. Not part of any verdict.
    cond_rows = []
    for label, d in (("development", dev), ("replication", rep), ("heldout_regime", opn)):
        sc = all_scores(d, [n for n in fixed_names], add_ref, models)
        for outcome in ("entry_compromise_rate", "conditional_collapse_rate"):
            m = d[outcome].notna().values
            y = d[outcome].values[m]
            for k in DESCRIPTIVE_SCORES + list(FEATURE_SETS):
                if k in sc:
                    cond_rows.append(dict(sample=label, outcome=outcome, score=k,
                                          n=int(m.sum()), c=concordance(sc[k][m], y)))
    pd.DataFrame(cond_rows).to_csv(p("decomposition_concordance.csv"), index=False)

    dev_scores = all_scores(dev, fixed_names, add_ref, {})
    aucs, dls = secondary_auc(dev, dev_scores, design)
    aucs.to_csv(p("secondary_auc.csv"), index=False)
    dls.to_csv(p("secondary_delong.csv"), index=False)
    six_auc = {k: v for k, v in dev_scores.items() if k in ["S_I", "S_A", "S_X", "S_T", "S_rem", "S_vis"]}
    ythr = (dev["trial_collapse_rate"].values >= design["secondary_threshold"]).astype(int)
    if 0 < ythr.sum() < len(ythr):
        bs = max(six_auc, key=lambda k: roc_auc_score(ythr, six_auc[k]))
        plot_roc(dev, dev_scores, [bs, "ADD", "F1", "FC"], design["secondary_threshold"], p("roc_secondary.png"))
    plot_ladder(ctab[ctab["sample"] == "replication"], p("concordance_ladder.png"))

    print(f"  interaction screen ({design['n_perm']} permutations per pair)...")
    inter = interaction_screen(dev, COUPLING_AXES, design["secondary_threshold"],
                               design["n_perm"], design["boot_seed"])
    inter.to_csv(p("interaction_screen.csv"), index=False)

    oat = os.path.join(run_dir, "oat_sweep.csv")
    if os.path.exists(oat):
        oat_profile_similarity(pd.read_csv(oat)).to_csv(p("oat_profile_similarity.csv"), index=False)

    write_summary(p("summary.md"), verdict, conf, ctab, status, fz, pre_ok, pre_diffs, design)
    files = sorted(f for f in os.listdir(run_dir) if f.startswith("validation_") or f == "oat_sweep.csv"
                   or f == "P4A_prespecification.json")
    manifest = dict(
        code_version=CODE_VERSION, code_sha256=code_hash(),
        generated=_dt.datetime.now().isoformat(timespec="seconds"),
        core_version=fz["core_version"], freeze_digest=fz["freeze_digest"], freeze_note=fz["note"],
        prespecification_matches=pre_ok, prespecification_differences=pre_diffs,
        citable=bool(fz["verified"] and pre_ok),
        python=sys.version.split()[0],
        packages={m: getattr(__import__(m), "__version__", "n/a")
                  for m in ("numpy", "pandas", "scipy", "sklearn", "matplotlib")},
        inputs={f: _sha256(os.path.join(run_dir, f)) for f in files},
        outputs=sorted(f for f in os.listdir(run_dir) if f.startswith("P4A_")) + ["P4A_manifest.json"])
    with open(p("manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print("\nVerdicts:")
    for _, r in verdict.iterrows():
        print(f"  {r.hypothesis:4s} {r.n_pass}/{r.n_comparisons}  {r.verdict}")
    print(f"Citable: {manifest['citable']}.  Outputs -> {run_dir}")
    return dict(verdict=verdict, confirmatory=conf)


# ############################################################################
# ENTRY POINT
# ############################################################################

MODES = [("SMOKE", "smoke", "pilot seeds, 40 scenarios x 20 trials, minutes, not for the paper"),
         ("ALL", "all", "full confirmatory run: generate, then analyse (hours)"),
         ("GENERATE", "generate", "generate or resume samples only"),
         ("ANALYSE", "analyse", "analyse an existing run directory")]


def choose_mode() -> str:
    print("Select mode:")
    for k, (label, _, desc) in enumerate(MODES, 1):
        print(f"  {k}  {label:9s} {desc}")
    while True:
        a = input("Enter number or name: ").strip().upper()
        for k, (label, mode, _) in enumerate(MODES, 1):
            if a in (str(k), label, mode.upper()):
                return mode
        print("  not recognised")


def choose_run_id(out_root: str) -> str:
    runs = sorted((d for d in os.listdir(out_root)
                   if os.path.exists(os.path.join(out_root, d, "P4A_prespecification.json")))
                  if os.path.isdir(out_root) else [])
    if not runs:
        raise SystemExit(f"No runs with a pre-specification record under {out_root}.")
    print("Existing runs:")
    for k, r in enumerate(runs, 1):
        print(f"  {k}  {r}")
    while True:
        a = input("Enter number or run id: ").strip()
        if a.isdigit() and 1 <= int(a) <= len(runs):
            return runs[int(a) - 1]
        if a in runs:
            return a
        print("  not recognised")


def main(argv=None):
    ap = argparse.ArgumentParser(description=CODE_VERSION)
    ap.add_argument("--mode", type=str.lower, choices=["smoke", "all", "generate", "analyse"])
    ap.add_argument("--run-id")
    ap.add_argument("--out-root")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--exploratory", action="store_true")
    a = ap.parse_args(argv)
    cfg = dict(RUN_CONFIG)
    for k in ("mode", "run_id", "out_root", "workers"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    if a.exploratory:
        cfg["exploratory"] = True
    mode = str(cfg["mode"]).lower()
    if mode == "ask":
        mode = choose_mode()
    if mode not in ("smoke", "all", "generate", "analyse"):
        raise SystemExit(f"Unknown mode {cfg['mode']!r}.")
    if mode == "analyse" and not cfg["run_id"]:
        cfg["run_id"] = choose_run_id(cfg["out_root"])

    print("=" * 70)
    print(f"  {CODE_VERSION}   mode={mode}")
    print("=" * 70)
    fz = freeze_status()
    print(f"  embedded cemt_core {fz['core_version']}: {fz['note']}")

    run_id = cfg["run_id"] or _dt.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_p4a_{mode}_v{VERSION}"
    run_dir = os.path.join(cfg["out_root"], run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"  run directory: {run_dir}")

    pre_path = os.path.join(run_dir, "P4A_prespecification.json")
    if mode in ("smoke", "all", "generate"):
        design = dict(DESIGN, **(SMOKE_DESIGN if mode == "smoke" else {}))
        if os.path.exists(pre_path):
            design = json.load(open(pre_path))["design"]     # resume under the saved design
        write_prespecification(run_dir, prespecification(design, fz))
        workers = cfg["workers"] or max(1, (os.cpu_count() or 2) - 2)
        n = (2 + len(design["replication_seeds"])) * design["n_scenarios"]
        print(f"  {n} scenarios x {design['n_trials']} trials, {workers} worker(s)")
        generate(design, run_dir, workers)
    if mode in ("smoke", "all", "analyse"):
        if not os.path.exists(pre_path):
            raise SystemExit(f"No pre-specification record in {run_dir}.")
        design = json.load(open(pre_path))["design"]
        pre_ok, diffs = check_prespecification(run_dir, prespecification(design, fz))
        if not pre_ok:
            msg = ("The current file differs from the run's pre-specification in: "
                   + ", ".join(sorted(diffs)) + ".")
            if not cfg["exploratory"]:
                raise SystemExit(msg + " Analysis stopped. Restore the pre-specified version, "
                                 "or set exploratory=True for a non-citable analysis.")
            print("  WARNING: " + msg + " Exploratory analysis; output is not citable.")
        analyse(run_dir, design, fz, pre_ok, diffs)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()