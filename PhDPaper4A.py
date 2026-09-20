# -*- coding: utf-8 -*-
"""
PhDPaper4A v11: Structure. Per-node compromise probability on cemt_core v0.8
==============================================================================

Implements "Paper 4A Specification v12: Structure" (20 September 2026).

The paper estimates the probability that each node in a system is compromised
given delivery from a compromised external source, as a function of structure
alone, and shows that the frozen cemt_core v0.8 model reproduces the imported
laws of propagation (mesh percolation, hub fragility, chain decay) when
execution is gated by the node's Cyber Triangle (I, X, A). Time and Adaptation
are switched off (Paper 3A's M_S); 4B and 4C re-enable them on the same saved
architectures.

One self-contained file. The frozen cemt_core v0.8 is embedded byte for byte
and hash-checked at load (digest 97dbae47); a mismatch aborts. Third-party
libraries: numpy, pandas, scipy only. Everything new is driver-level or
analysis-level: no mechanic of the frozen core changes.

Entry (specification Sections 5, 6.3, 8): every architecture carries external
source nodes (relay, update, registry). A run pins entry to one source, which
is compromised at step 0 by assumption (entry_certain) and delivers on every
relation it supplies; each receiver's triangle decides whether the payload
executes there. A node cannot pass anything on until it has executed;
carriage by uncompromised estate nodes is a declared v0.9 candidate.

Hypotheses (Section 10; supported only if every criterion holds on 4201, 4202
and 4301 separately; both replications passing with a bounded held-out failure
is "supported under replication, not generalised"):
  H1 mesh   percolation threshold gated by f: (a) sigmoid rise per f, (b) the
            midpoint density falls with f, (c) midpoint x f roughly constant
            (CV < 0.25), (d) plateau within 0.05 of f.
  H2 star   deleting the hub costs more than deleting five leaves, both forms,
            every f and n (paired CI lower bound > 0).
  H3 chain  max positive adjacent-depth step and max standardised deviation
            both inside their simulated 95% critical values, and within 0.05
            of prediction at f = 1.
  H4 pos    Firth binomial regression of p_i on gated distance, log(1+redundancy)
            and their interaction: distance < 0, redundancy > 0, interaction > 0,
            fitted separately on each evaluation sample.
  H5 shape  (a) matched-count blocks (same 59 connections rearranged): chain
            below star by >= 0.05 and below mesh (CI > 0); (b) representative
            blocks (same nodes, gate and external entry points; each shape at
            its natural connectivity): chain below star by >= 0.05 and below
            mesh at mean degree 6 and 12 (CI > 0). The entry-share curve of Y
            per shape is an exploratory output.
  H6 gate   gated model beats the eligible-subgraph ungated model by >= 0.02
            concordance, paired CI lower bound > 0, fitted on development only;
            the whole-graph ungated rival is reported as secondary.
Implementation checks IC1..IC8 (Section 11) abort the run on failure.

Modes: PILOT (pilot seeds, reduced budget, broad screen, no verdicts), TIME
(budget estimate), ALL (generate then analyse), GENERATE (resumable),
ANALYSE (existing run). Quick read: P4A_summary.md in the run directory.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import hashlib
import io
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.stats import qmc

# ============================================================================
# VERSION (single source of truth for code, specification and tag)
# ============================================================================
VERSION = 12
CODE_VERSION = f"PhDPaper4A v{VERSION} (2026-09-20)"
SPEC_VERSION = "Paper 4A Specification v12: Structure (2026-09-20)"

# >>> RUN_CONFIG (excluded from the analysis-code hash)
RUN_CONFIG = dict(
    mode="ask",                # ask | pilot | all | generate | analyse | time
    out_root=r"C:\Users\Paul.Guckian\Documents\Phd\P4A",
    run_id=None,               # None = new run; set to resume or analyse
    workers=None,              # None = cpu_count - 2
    exploratory=False,         # allow analysis despite a record mismatch (not citable)
    write_runs=True,           # per-run records to runs.csv.gz (large)
    write_specs=True,          # one scenario file per architecture in specs/
    v8_run_dir=None,           # optional: path of the v8 run for the exploratory reanalysis
)
# <<< RUN_CONFIG

# ============================================================================
# PRE-SPECIFICATION (recorded in run_record.json and enforced at ANALYSE)
# ============================================================================
DESIGN = dict(
    seeds=dict(development=4101, replication=[4201, 4202], heldout=4301),
    pilot_seeds=dict(development=231, replication=[543], heldout=311),
    runs_per_arch=1000, pilot_runs_per_arch=100,
    arch_per_cell=10, pilot_arch_per_cell=10,         # the pilot uses the full design at fewer runs
    matched_blocks=20, pilot_matched_blocks=20,
    rep_blocks=10, pilot_rep_blocks=10,                # representative-shape blocks per (f, entry share)
    entry_shares=[0.01, 0.05, 0.10, 0.15, 0.20],       # external entry points as a share of estate nodes
    mesh_degrees=[6, 12],                              # representative mesh mean degrees
    h3_family_cells=10,                                # 2 chain forms x 5 f: band level 1 - 0.05/10 per cell
    mixed_per_sample=400, pilot_mixed_per_sample=400,
    n_boot=2000, pilot_n_boot=200, boot_seed=20260920, model_seed=20260920,
    f_levels=[0.2, 0.4, 0.6, 0.8, 1.0],
    mesh_grid_exponents=[-2 + 5 * j / 11 for j in range(12)],   # d_j = min(0.95, d*_{N,f} 2^(-2+5j/11))
    mesh_density_cap=0.95,
    star_n=[10, 20, 40, 59], star_leaf_removals=[1, 5],
    chain_depth=20,
    n_canonical=60, n_canonical_heldout=120,
    h1_cv_max=0.25, h1_plateau_tol=0.05, h3_f1_tol=0.05,
    h5_min_delta=0.05, h6_min_delta_c=0.02, alpha=0.05,
    docker_per_shape=3, docker_n=30, docker_f=0.6, docker_seeds=20,
    redundancy_cap=5,
    h4_ridge_grid=[0.0, 0.01, 0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0], h4_cv_folds=5,
)

# Mixed-architecture factors (Section 6.2): name -> (low, high, kind)
FACTORS_DEV = {
    "n_nodes": (40, 120, "int"),
    "p_I": (0.20, 0.90, "lin"), "p_X": (0.20, 0.90, "lin"), "p_A": (0.20, 0.90, "lin"),
    "colocation": (0.0, 0.8, "lin"),
    "conn_density": (0.01, 0.15, "lin"),
    "chan_density": (0.005, 0.05, "lin"),
    "conf_count": (0, 3, "int"), "conf_membership": (0.05, 0.50, "lin"),
    "dir_count": (0, 3, "int"), "dir_fanout": (0.05, 0.60, "lin"),
    "share_external": (0.0, 1.0, "lin"),
    "bundle_chan": (0.0, 1.0, "lin"), "bundle_dir": (0.0, 1.0, "lin"),
    "relay_share": (0.20, 1.00, "lin"),
    "registry_targets": (1, 5, "int"),
}
FACTORS_HELDOUT = dict(FACTORS_DEV, n_nodes=(120, 200, "int"), conn_density=(0.15, 0.30, "lin"),
                       chan_density=(0.05, 0.10, "lin"), dir_fanout=(0.60, 0.95, "lin"))
SOURCE_TYPES = ["relay", "update", "registry"]

# Fixed in every architecture (Section 5)
FIXED = dict(threat_capability=1.0, authority_rate=0.55, execution_rate=0.55,
             control_variance=0.0, beta_conn=0.12, dependency_factor=2.0,
             execution_drift_boost=0.0, control_plane_takeover_rate=0.08, authority_boost=2.0,
             synchrony=0.5, drift_increment=0.0, exfil_dwell_steps=5, max_steps=50,
             latency_steps=0, entry_certain=True,
             phantom_cp_rate=0.0, phantom_channel_rate=0.0, phantom_dep_rate=0.0,
             ablation=dict(structure=True, time=False, adaptation=False),
             impact_lognormal=dict(mean=0.0, sigma=1.0))

# Per-node predictor sets (Section 7.1, 10, 13)
GATED = ["gated_distance", "log_redundancy", "dist_x_red"]
UNGATED = ["ungated_distance", "log_ungated_redundancy", "udist_x_ured"]
INDUCED = ["induced_distance", "log_induced_redundancy", "idist_x_ired"]
NODE_MEASURES = ["I_native", "X_native", "A_native", "I_sup", "X_sup", "A_sup",
                 "complete_conn", "complete_chan", "eligible", "n_ways", "plane_member", "touchpoint",
                 "deliv_conn", "deliv_chan", "deliv_dir",
                 "gated_distance", "ungated_distance", "induced_distance",
                 "redundancy", "ungated_redundancy", "induced_redundancy",
                 "hub_member", "hub_max_fanout", "mech_reachable", "boundary_X", "boundary_A",
                 "cut_I", "cut_X", "cut_A"]
ARCH_MEASURES = ["source_degree", "f_IXA", "f_XA", "f_A", "f_XA_native", "f_A_native", "complete_count",
                 "redundancy_mean", "hub_concentration", "centralisation", "chain_depth",
                 "cut_conn", "cut_chan", "cut_conf", "cut_dir",
                 "fanout_max_conn", "fanout_max_chan", "fanout_max_conf", "fanout_max_dir",
                 "boundary_share", "deg_mean", "deg_max", "betw_mean", "betw_max",
                 "reach_ungated", "n_scc", "path_mean"]

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
        "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"
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
    try:
        import yaml  # noqa: F401  (used only by the core's file loaders)
    except ImportError:
        stub = types.ModuleType("yaml")
        def _no_yaml(*a, **k):
            raise RuntimeError("PyYAML is not installed; the core's YAML loaders are unavailable")
        stub.safe_load = stub.safe_dump = stub.load = stub.dump = _no_yaml
        sys.modules["yaml"] = stub
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
# PART 1: DRIVER (architectures, measures, runs)
# ############################################################################

class ICFailure(RuntimeError):
    """An implementation check failed inside a worker; generation stops with the message."""


_CORE = None


def _core():
    global _CORE
    if _CORE is None:
        pkg, report = _load_embedded_core()
        sp = sys.modules["cemt_core.spec"]
        _CORE = SimpleNamespace(
            build_network=pkg.build_network, run_one_trial=pkg.run_one_trial,
            CORE_VERSION=pkg.CORE_VERSION, sp=sp, RC=sp.RelationClass, Cond=sp.Condition,
            RelationTable=sys.modules["cemt_core.relations"].RelationTable, hash_report=report)
    return _CORE


def _guard_validate(sp) -> None:
    """IC8: ScenarioSpec.validate is never called on any built spec (latent DEPENDENCY reference)."""
    def _never(self, *a, **k):
        raise SystemExit("IC8 failed: ScenarioSpec.validate was called; it references RelationClass.DEPENDENCY")
    sp.ScenarioSpec.validate = _never


def ic6_fixtures() -> dict:
    """IC6 fixtures: two paths sharing one internal vertex give redundancy 1 after node-splitting;
    Cut values on single-supply, alternative-supply and shared-upstream fixtures."""
    # fixture 1: roots 0 and 1 both lead through vertex 2 to target 3
    adj = [[2], [2], [3], []]
    r = _disjoint_paths(adj, [0, 1], 3, 4, 5)
    # fixture 2: roots 0 and 1 lead separately to target 3 through 2 and 4
    adj2 = [[2], [4], [3], [], [3]]
    r2 = _disjoint_paths(adj2, [0, 1], 3, 5, 5)
    # cut fixtures on the frozen relation table
    c, sp, RC, Cond = _core(), _core().sp, _core().RC, _core().Cond
    gov = [sp.Governance("estate", True), sp.Governance("vendor", False)]
    def mk(rels):
        nodes = [_node(sp, "a", "estate", True, True, True), _node(sp, "b", "estate", True, True, True),
                 _node(sp, "t", "estate", False, True, True)]
        spec = sp.ScenarioSpec(id="fx", description="", governance=gov, nodes=nodes, relations=rels, seed=1)
        table = c.RelationTable(spec)
        cs = table.cut_set("t")
        return cs.get(Cond.INTERFACE)
    single = mk([sp.Relation("a", "t", Cond.INTERFACE, RC.CONNECTION)])
    alt = mk([sp.Relation("a", "t", Cond.INTERFACE, RC.CONNECTION), sp.Relation("b", "t", Cond.INTERFACE, RC.CONNECTION)])
    ok = (r == 1 and r2 == 2 and single is not None and single[0] == "a" and alt is None)
    return dict(passed=bool(ok), shared_vertex_redundancy=r, separate_paths_redundancy=r2,
                single_supply_cut=str(single), alternative_supply_cut=str(alt))


def freeze_status() -> dict:
    c = _core()
    _guard_validate(c.sp)
    ok = all(c.hash_report.values())
    bad = [k for k, v in c.hash_report.items() if not v]
    return dict(core_version=c.CORE_VERSION, freeze_digest=EMBEDDED_SET_DIGEST if ok else None,
                verified=ok, note=("embedded core matches freeze record" if ok else
                                   "EMBEDDED CORE DIFFERS from freeze record: " + ", ".join(bad)))


# ---- specification assembly -------------------------------------------------

def _node(sp, nid, gov, I, X, A, impact=1.0, visible=True):
    return sp.Node(id=nid, governance=gov, interface=bool(I), execution_pathway=bool(X),
                   authority=bool(A), visible=visible, impact=float(impact))


def _rel(c, s, r, cond, rc, group=None):
    return c.sp.Relation(s, r, cond, rc, group=group) if group else c.sp.Relation(s, r, cond, rc)


def make_spec(arch_id: str, nodes, rels, seed: int, deterministic: bool = False):
    c, sp = _core(), _core().sp
    ab = FIXED["ablation"]
    return sp.ScenarioSpec(
        id=arch_id, description=f"P4A v{VERSION}",
        governance=[sp.Governance("estate", True), sp.Governance("vendor", False)],
        nodes=nodes, relations=rels, entry_certain=FIXED["entry_certain"], seed=seed,
        deterministic=deterministic,
        rates=sp.RateSpec(threat_capability=FIXED["threat_capability"],
                          authority_rate=FIXED["authority_rate"], execution_rate=FIXED["execution_rate"],
                          control_variance=FIXED["control_variance"], beta_conn=FIXED["beta_conn"],
                          synchrony=FIXED["synchrony"], dependency_factor=FIXED["dependency_factor"],
                          execution_drift_boost=FIXED["execution_drift_boost"],
                          control_plane_takeover_rate=FIXED["control_plane_takeover_rate"],
                          authority_boost=FIXED["authority_boost"],
                          phantom_cp_rate=FIXED["phantom_cp_rate"],
                          phantom_channel_rate=FIXED["phantom_channel_rate"],
                          phantom_dep_rate=FIXED["phantom_dep_rate"]),
        time=sp.TimeSpec(latency_steps=FIXED["latency_steps"], drift_increment=FIXED["drift_increment"],
                         exfil_dwell_steps=FIXED["exfil_dwell_steps"], max_steps=FIXED["max_steps"]),
        adaptation=sp.AdaptationSpec(remediation_capability=0.0, synchrony=FIXED["synchrony"],
                                     threshold=0.0, tau_a=0.3, simple_policy=True),
        ablation=sp.AblationSpec(structure=ab["structure"], time=ab["time"], adaptation=ab["adaptation"]))


def _impacts(rng, n):
    p = FIXED["impact_lognormal"]
    return rng.lognormal(p["mean"], p["sigma"], n)


def _exact_subset(rng, n, f, forced=()):
    """Exactly round(f*n) nodes hold A; forced indices always do."""
    k = int(round(f * n))
    A = np.zeros(n, dtype=bool)
    forced = list(forced)
    A[forced] = True
    others = [i for i in range(n) if i not in forced]
    need = max(0, k - len(forced))
    if need and others:
        A[rng.choice(others, min(need, len(others)), replace=False)] = True
    return A


def _source(sp, k=0):
    return _node(sp, f"src{k}", "vendor", True, True, True, impact=0.0, visible=False)


# ---- canonical shapes (Section 6.1) -----------------------------------------

SCALE_TABLE = {120: {10: 20, 20: 40, 40: 80, 59: 119, 1: 2, 5: 10}}   # Section 6.1 held-out dimensions


def scale60(N: int, v: int) -> int:
    """Canonical dimension v at N = 60 mapped to N: the stated table for N = 120, else proportional."""
    if N == 60:
        return int(v)
    if N in SCALE_TABLE and v in SCALE_TABLE[N]:
        return SCALE_TABLE[N][v]
    return max(1, int(round(v * (N - 1) / 59.0)))


def mesh_densities(N: int, f: float) -> List[float]:
    """Section 6.1: d_j = min(0.95, d*_{N,f} 2^(-2+5j/11)), with the nominal-gate d*."""
    pi = FIXED["beta_conn"] * FIXED["execution_rate"] * FIXED["authority_rate"]
    T_H = 1 - (1 - pi) ** FIXED["max_steps"]
    d_star = 1.0 / ((N - 1) * f * T_H)
    return [min(DESIGN["mesh_density_cap"], d_star * 2 ** e) for e in DESIGN["mesh_grid_exponents"]]


def build_canonical(task: dict):
    """Returns (spec, node_meta, arch_meta). Exactly round(f*N) estate nodes hold A (IC4);
    nothing is forced complete. The source is src0, outside the boundary."""
    c, sp, RC, Cond = _core(), _core().sp, _core().RC, _core().Cond
    rng = np.random.default_rng(task["seed"])
    N, f, shape = task["N"], task["f"], task["shape"]
    ids = [f"n{k}" for k in range(N)]
    I = rng.random(N) < 0.5
    X = np.ones(N, dtype=bool)
    imp = _impacts(rng, N)
    nodes, rels, meta = [], [], {}
    d = task.get("d", 0.0)
    n = task.get("n", 0)
    variant = task.get("variant", "intact")
    removed = set()
    if shape in ("star_cp", "star_conn"):
        # Section 6.1: the entry leaf and the hub or controller are always among the A holders
        hub = 0
        leaves = list(rng.choice(np.arange(1, N), n, replace=False))
        entry_leaf = int(rng.choice(leaves))
        A = _exact_subset(rng, N, f, [hub, entry_leaf])
    else:
        A = _exact_subset(rng, N, f, [])
    if shape == "mesh":
        V = N + 1                      # the source is a vertex of one G(V, d) mesh: independent edge draws
        iu, ju = np.triu_indices(V, 1)
        keep = rng.random(len(iu)) < d
        pairs = sorted(zip(iu[keep].tolist(), ju[keep].tolist()))
        for i, j in pairs:
            a_, b_ = ("src0" if i == N else ids[i]), ("src0" if j == N else ids[j])
            rels.append(_rel(c, a_, b_, Cond.INTERFACE, RC.CONNECTION))
            if i == N or j == N:
                meta[b_ if i == N else a_] = "touchpoint"
    elif shape in ("star_cp", "star_conn", "star_hubsource"):
        if shape == "star_hubsource":
            hub = 0
            leaves = list(rng.choice(np.arange(1, N), n, replace=False))
            entry_leaf = int(rng.choice(leaves))
        I[hub] = False
        if variant == "hub_removed":
            removed.add(hub)
        elif variant.startswith("leaves_removed_"):
            k = int(variant.split("_")[-1])
            cand = [l for l in leaves if l != entry_leaf]
            removed.update(int(x) for x in rng.choice(cand, min(k, len(cand)), replace=False))
        keep_leaves = [l for l in leaves if l not in removed]
        if shape == "star_hubsource":
            for l in keep_leaves:
                rels.append(_rel(c, "src0", ids[l], Cond.INTERFACE, RC.CONTROL_PLANE_DIRECTING, group="plane0"))
            removed.add(hub)
        else:
            if hub not in removed:
                for l in keep_leaves:
                    if shape == "star_cp":
                        rels.append(_rel(c, ids[hub], ids[l], Cond.INTERFACE, RC.CONTROL_PLANE_DIRECTING, group="plane0"))
                    else:
                        rels.append(_rel(c, ids[hub], ids[l], Cond.INTERFACE, RC.CONNECTION))
            rels.append(_rel(c, "src0", ids[entry_leaf], Cond.INTERFACE, RC.CONNECTION))
        meta[ids[hub]] = "hub"
        for l in leaves:
            meta[ids[l]] = "entry_leaf" if l == entry_leaf else "leaf"
    elif shape in ("chain_chan", "chain_conn"):
        depth = task.get("depth", DESIGN["chain_depth"] if N == 60 else (40 if N == 120 else scale60(N, DESIGN["chain_depth"])))
        cls = RC.CHANNEL if shape == "chain_chan" else RC.CONNECTION
        cond = Cond.EXECUTION_PATHWAY if shape == "chain_chan" else Cond.INTERFACE
        if task.get("sixth") is not None:
            # Docker chain pair: the base holds A at the sixth node, the negative control does not;
            # the total A count is preserved by swapping with a node outside the chain
            want = bool(task["sixth"])
            if A[5] != want:
                outside = [k for k in range(depth, N) if A[k] != want]
                if outside:
                    A[int(rng.choice(outside))] = not want
                A[5] = want
        rels.append(_rel(c, "src0", ids[0], cond, cls))
        for k in range(depth - 1):
            rels.append(_rel(c, ids[k], ids[k + 1], cond, cls))
        for k in range(depth):
            meta[ids[k]] = f"depth{k + 1}"
        I[:depth] = False
    elif shape in ("matched_mesh", "matched_star", "matched_chain"):
        R = N - 1
        if shape == "matched_mesh":
            pairs = set()
            while len(pairs) < R:
                i, j = rng.integers(0, N, 2)
                if i != j:
                    pairs.add((min(i, j), max(i, j)))
            edges = sorted(pairs)
        elif shape == "matched_star":
            edges = [(0, j) for j in range(1, N)]
        else:
            edges = [(k, k + 1) for k in range(N - 1)]
        for i, j in edges:
            rels.append(_rel(c, ids[i], ids[j], Cond.INTERFACE, RC.CONNECTION))
        rels.append(_rel(c, "src0", ids[0], Cond.INTERFACE, RC.CONNECTION))
        meta[ids[0]] = "touchpoint"
    else:
        raise ValueError(shape)
    for k in range(N):
        if k in removed:
            continue
        nodes.append(_node(sp, ids[k], "estate", I[k], X[k], A[k], imp[k]))
    nodes.append(_source(sp))
    spec = make_spec(task["arch_id"], nodes, rels, task["seed"])
    arch_meta = dict(f_target=f, d=d, n=n, variant=variant, sources="src0", source_types="canonical",
                     N_target=N, n_A_assigned=int(A.sum()))
    return spec, meta, arch_meta


# ---- representative-shape blocks (Section 6.1, H5b) --------------------------------------

def build_representative(task: dict):
    """Same nodes, A holders, impacts, gate draws and external entry points in every architecture
    of a block; the shape sets the internal connectivity. Native Interface marks exactly the
    entry points, and the source connects to exactly those."""
    c, sp, RC, Cond = _core(), _core().sp, _core().RC, _core().Cond
    rng = np.random.default_rng(task["seed"])                       # block-level draws
    N, f, shape = task["N"], task["f"], task["shape"]
    ids = [f"n{k}" for k in range(N)]
    imp = _impacts(rng, N)
    A = _exact_subset(rng, N, f, [])
    n_entry = max(1, int(round(task["entry_share"] * N)))
    entry = sorted(int(x) for x in rng.choice(N, n_entry, replace=False))
    non_entry = [k for k in range(N) if k not in entry]
    hub = int(rng.choice(non_entry)) if non_entry else 0
    I = np.zeros(N, dtype=bool); I[entry] = True
    X = np.ones(N, dtype=bool)
    srng = np.random.default_rng(task["seed"] * 7919 + {"rep_mesh6": 1, "rep_mesh12": 2, "rep_star": 3, "rep_chain": 4}[shape])
    rels, meta = [], {}
    for k in entry:
        rels.append(_rel(c, "src0", ids[k], Cond.INTERFACE, RC.CONNECTION))
        meta[ids[k]] = "entry"
    if shape.startswith("rep_mesh"):
        degree = int(shape[len("rep_mesh"):])
        p = min(1.0, degree / (N - 1))
        iu, ju = np.triu_indices(N, 1)
        keep = srng.random(len(iu)) < p
        for i, j in zip(iu[keep].tolist(), ju[keep].tolist()):
            rels.append(_rel(c, ids[i], ids[j], Cond.INTERFACE, RC.CONNECTION))
    elif shape == "rep_star":
        for j in range(N):
            if j != hub:
                rels.append(_rel(c, ids[hub], ids[j], Cond.INTERFACE, RC.CONNECTION))
        meta[ids[hub]] = "hub"
    else:
        order = srng.permutation(N)
        for a_, b_ in zip(order[:-1], order[1:]):
            rels.append(_rel(c, ids[int(a_)], ids[int(b_)], Cond.INTERFACE, RC.CONNECTION))
    nodes = [_node(sp, ids[k], "estate", I[k], X[k], A[k], imp[k]) for k in range(N)] + [_source(sp)]
    spec = make_spec(task["arch_id"], nodes, rels, task["seed"])
    return spec, meta, dict(f_target=f, d=0.0, n=n_entry, variant=shape, sources="src0", source_types="canonical",
                            N_target=N, n_A_assigned=int(A.sum()), entry_share=task["entry_share"])


# ---- mixed architectures (Section 6.2) -------------------------------------

def draw_factors(n: int, ranges: dict, seed: int) -> List[dict]:
    names = list(ranges)
    u = qmc.LatinHypercube(d=len(names), seed=seed).random(n)
    out = []
    for row in u:
        f = {}
        for name, v in zip(names, row):
            lo, hi, kind = ranges[name]
            f[name] = float(lo + v * (hi - lo)) if kind == "lin" else int(lo + math.floor(v * (hi - lo + 1)))
        out.append(f)
    return out


def build_mixed(task: dict):
    c, sp, RC, Cond = _core(), _core().sp, _core().RC, _core().Cond
    f = task["factors"]
    rng = np.random.default_rng(task["seed"])
    N = int(f["n_nodes"])
    ids = [f"n{k}" for k in range(N)]
    rho = f["colocation"]
    common = rng.standard_normal(N)
    pres = {}
    for cond, rate in (("I", f["p_I"]), ("X", f["p_X"]), ("A", f["p_A"])):
        z = np.sqrt(rho) * common + np.sqrt(1 - rho) * rng.standard_normal(N)
        pres[cond] = z < stats.norm.ppf(rate)
    imp = _impacts(rng, N)
    nodes = [_node(sp, ids[k], "estate", pres["I"][k], pres["X"][k], pres["A"][k], imp[k]) for k in range(N)]
    rels, meta = [], {}
    conn_pairs = set()
    m = int(round(f["conn_density"] * N * (N - 1) / 2))
    while len(conn_pairs) < m:
        i, j = rng.integers(0, N, 2)
        if i != j:
            conn_pairs.add((min(i, j), max(i, j)))
    for i, j in sorted(conn_pairs):
        rels.append(_rel(c, ids[i], ids[j], Cond.INTERFACE, RC.CONNECTION))
    chan = set()
    m = int(round(f["chan_density"] * N * (N - 1)))
    while len(chan) < m:
        i, j = rng.integers(0, N, 2)
        if i != j:
            chan.add((int(i), int(j)))
    for i, j in conn_pairs:
        if rng.random() < f["bundle_chan"]:
            chan.add((i, j) if rng.random() < 0.5 else (j, i))
    for i, j in sorted(chan):
        rels.append(_rel(c, ids[i], ids[j], Cond.EXECUTION_PATHWAY, RC.CHANNEL))
    for cls, ck, sk in (("conf", "conf_count", "conf_membership"), ("dir", "dir_count", "dir_fanout")):
        for k in range(int(f[ck])):
            ext = rng.random() < f["share_external"]
            src = f"{cls}{k}"
            nodes.append(_node(sp, src, "vendor" if ext else "estate", True, True, True, 0.0, visible=not ext))
            mem = rng.choice(N, max(1, int(round(f[sk] * N))), replace=False)
            for j in mem:
                if cls == "conf":
                    rels.append(_rel(c, src, ids[j], Cond.AUTHORITY, RC.CONTROL_PLANE_CONFERRING, group=src))
                else:
                    rels.append(_rel(c, src, ids[j], Cond.INTERFACE, RC.CONTROL_PLANE_DIRECTING, group=src))
                    if rng.random() < f["bundle_dir"]:
                        rels.append(_rel(c, src, ids[j], Cond.INTERFACE, RC.CONNECTION))
    # sources (Section 6.3): 1 + (a mod 3) sources; the sample seed fixes one base permutation
    # of the three types, rotated by a mod 3, first n_src taken (without replacement)
    a_idx = int(task["index"])
    n_src = 1 + a_idx % 3
    base = list(np.random.default_rng(task["meta"]["sample_seed"] * 31 + 5).permutation(SOURCE_TYPES))
    rot = a_idx % 3
    types = (base[rot:] + base[:rot])[:n_src]
    src_ids = []
    for k, typ in enumerate(types):
        sid = f"src{k}"
        src_ids.append(sid)
        nodes.append(_source(sp, k))
        if typ == "relay":
            tgt = rng.choice(N, max(1, int(round(f["relay_share"] * N))), replace=False)
            for j in tgt:
                rels.append(_rel(c, sid, ids[j], Cond.INTERFACE, RC.CONNECTION))
        elif typ == "update":
            tgt = rng.choice(N, max(1, int(round(f["dir_fanout"] * N))), replace=False)
            for j in tgt:
                rels.append(_rel(c, sid, ids[j], Cond.INTERFACE, RC.CONTROL_PLANE_DIRECTING, group=f"{sid}plane"))
        else:
            tgt = rng.choice(N, min(N, int(f["registry_targets"])), replace=False)
            for j in tgt:
                rels.append(_rel(c, sid, ids[j], Cond.EXECUTION_PATHWAY, RC.CHANNEL))
        meta[sid] = typ
    spec = make_spec(task["arch_id"], nodes, rels, task["seed"])
    return spec, meta, dict(sources=";".join(src_ids), source_types=";".join(types))


# ---- static measures (Section 7) --------------------------------------------

def _bfs_multi(adj, roots, n):
    dist = np.full(n, -1)
    q = []
    for r in roots:
        if dist[r] < 0:
            dist[r] = 0
            q.append(r)
    for u in q:
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _disjoint_paths(adj_out, roots, target, n, cap):
    """Vertex-disjoint paths from any root to target, capped (Edmonds-Karp on
    the node-split graph, unit capacities)."""
    # node split: in = 2v, out = 2v+1 ; super source S = 2n, sink = target in
    S = 2 * n
    caps = {}
    def add(u, v):
        caps.setdefault(u, {})[v] = caps.get(u, {}).get(v, 0) + 1
        caps.setdefault(v, {}).setdefault(u, 0)
    for v in range(n):
        add(2 * v, 2 * v + 1)
    for u in range(n):
        for v in adj_out[u]:
            add(2 * u + 1, 2 * v)
    for r in roots:
        add(S, 2 * r)
    sink = 2 * target
    flow = 0
    while flow < cap:
        parent = {S: None}
        q = [S]
        found = False
        for u in q:
            for v, cp in caps.get(u, {}).items():
                if cp > 0 and v not in parent:
                    parent[v] = u
                    if v == sink:
                        found = True
                        break
                    q.append(v)
            if found:
                break
        if not found:
            break
        v = sink
        while parent[v] is not None:
            u = parent[v]
            caps[u][v] -= 1
            caps[v][u] += 1
            v = u
        flow += 1
    return flow


def _dominators(adj, roots, n):
    """Dominator sets from a super-source over the graph (Cooper, Harvey and Kennedy iteration).
    Returns, for each vertex, the set of vertices on every path from the roots to it (excluding
    itself), or None if unreachable."""
    S = n
    adj2 = [list(a) for a in adj] + [list(roots)]
    order, seen = [], [False] * (n + 1)
    stack = [(S, 0)]
    seen[S] = True
    while stack:
        v, i = stack[-1]
        if i < len(adj2[v]):
            stack[-1] = (v, i + 1)
            w = adj2[v][i]
            if not seen[w]:
                seen[w] = True
                stack.append((w, 0))
        else:
            stack.pop()
            order.append(v)
    rpo = order[::-1]
    pos = {v: k for k, v in enumerate(rpo)}
    preds = [[] for _ in range(n + 1)]
    for u in range(n + 1):
        for v in adj2[u]:
            if seen[v]:
                preds[v].append(u)
    idom = [-1] * (n + 1)
    idom[S] = S
    def intersect(a, b):
        while a != b:
            while pos[a] > pos[b]:
                a = idom[a]
            while pos[b] > pos[a]:
                b = idom[b]
        return a
    changed = True
    while changed:
        changed = False
        for v in rpo:
            if v == S:
                continue
            new = None
            for p in preds[v]:
                if idom[p] != -1:
                    new = p if new is None else intersect(p, new)
            if new is not None and idom[v] != new:
                idom[v] = new
                changed = True
    out = [None] * n
    for v in range(n):
        if not seen[v] or idom[v] == -1:
            continue
        d, cur = set(), idom[v]
        while cur != S:
            d.add(cur)
            cur = idom[cur]
        out[v] = d
    return out


def _betweenness_directed(adj, n):
    C = np.zeros(n)
    for s in range(n):
        S, P, sigma, dist = [], [[] for _ in range(n)], np.zeros(n), np.full(n, -1)
        sigma[s], dist[s] = 1, 0
        q = [s]
        for v in q:
            S.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)
        delta = np.zeros(n)
        for w in reversed(S):
            for v in P[w]:
                delta[v] += sigma[v] / sigma[w] * (1 + delta[w])
            if w != s:
                C[w] += delta[w]
    return C


def _scc_count(adj, n):
    index, low, onst, st, idx, count = [-1] * n, [0] * n, [False] * n, [], 0, 0
    for root in range(n):
        if index[root] >= 0:
            continue
        work = [(root, 0)]
        index[root] = low[root] = idx; idx += 1
        st.append(root); onst[root] = True
        while work:
            v, i = work[-1]
            if i < len(adj[v]):
                work[-1] = (v, i + 1)
                w = adj[v][i]
                if index[w] < 0:
                    index[w] = low[w] = idx; idx += 1
                    st.append(w); onst[w] = True
                    work.append((w, 0))
                elif onst[w]:
                    low[v] = min(low[v], index[w])
            else:
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[v])
                if low[v] == index[v]:
                    count += 1
                    while True:
                        w = st.pop(); onst[w] = False
                        if w == v:
                            break
    return count


def measure(spec, sources: List[str]) -> Tuple[Dict[str, dict], Dict[str, Dict[str, dict]], dict]:
    """Per-node static measures, per-source per-node measures (source-specific
    distance from the first internal touchpoints, redundancy, reachable hub
    membership) and per-architecture measures, all from the frozen RelationTable."""
    c = _core()
    RC, Cond = c.RC, c.Cond
    table = c.RelationTable(spec)
    gov_ctl = {g.id: g.defender_controlled for g in spec.governance}
    ids = [n.id for n in spec.nodes]
    idx = {nid: k for k, nid in enumerate(ids)}
    n_all = len(ids)
    ext = np.array([not gov_ctl[n.governance] for n in spec.nodes])
    est = [k for k in range(n_all) if not ext[k]]
    N = max(len(est), 1)
    nat = {n.id: (n.interface, n.execution_pathway, n.authority) for n in spec.nodes}
    sup: Dict[str, Dict] = {}
    fan: Dict[Tuple[str, str], set] = {}
    n_rows = ext_rows = 0
    for r in table.rows:
        sup.setdefault(r.receiver, {}).setdefault(r.condition, {}).setdefault(r.supplier, r.relation_class)
        if r.supplier != r.receiver:
            n_rows += 1
            ext_rows += ext[idx[r.supplier]]
            fan.setdefault((r.supplier, r.relation_class.value), set()).add(r.receiver)
    def S(nid, cnd):
        return sup.get(nid, {}).get(cnd, {})
    CI, CX, CA = Cond.INTERFACE, Cond.EXECUTION_PATHWAY, Cond.AUTHORITY
    supI = {nid: len(S(nid, CI)) > 0 for nid in ids}
    supX = {nid: len(S(nid, CX)) > 0 for nid in ids}
    supA = {nid: len(S(nid, CA)) > 0 for nid in ids}
    adj_g = [[] for _ in ids]
    adj_u = [[] for _ in ids]
    deliv = {nid: {"conn": 0, "chan": 0, "dir": 0} for nid in ids}
    planes: Dict[str, Tuple[str, List[str]]] = {}
    in_class: Dict[str, set] = {nid: set() for nid in ids}
    for r in table.rows:
        if r.supplier == r.receiver:
            continue
        u, v = idx[r.supplier], idx[r.receiver]
        if r.relation_class == RC.CONNECTION:
            ok = supX[r.receiver] and supA[r.receiver]; deliv[r.receiver]["conn"] += 1; in_class[r.receiver].add("conn")
        elif r.relation_class == RC.CHANNEL:
            ok = supA[r.receiver]; deliv[r.receiver]["chan"] += 1; in_class[r.receiver].add("chan")
        elif r.relation_class == RC.CONTROL_PLANE_DIRECTING:
            ok = supX[r.receiver] and supA[r.receiver]; deliv[r.receiver]["dir"] += 1; in_class[r.receiver].add("dir")
            g = getattr(r, "group", None) or r.supplier
            planes.setdefault(g, (r.supplier, []))[1].append(r.receiver)
        else:
            continue
        adj_u[u].append(v)
        if ok:
            adj_g[u].append(v)
    # membership of a directing plane is one hop from its controller only (Section 7.1); member
    # takeover is a Layer 3 rate and is not represented as reach in the measures. For the IC2
    # ceiling the eligible union uses the mechanism-complete reach, which includes takeover.
    adj_m = [list(a_) for a_ in adj_u]
    for g, (ctl, mem) in planes.items():
        for a_ in mem:
            for b_ in mem:
                if a_ != b_:
                    adj_m[idx[a_]].append(idx[b_])
    adj_m = [sorted(set(a_)) for a_ in adj_m]
    adj_g = [sorted(set(a_)) for a_ in adj_g]
    adj_u = [sorted(set(a_)) for a_ in adj_u]
    cap = DESIGN["redundancy_cap"]
    est_ids = [ids[k] for k in est]
    eligible = {nid: int((("conn" in in_class[nid] or "dir" in in_class[nid]) and supX[nid] and supA[nid])
                         or ("chan" in in_class[nid] and supA[nid])) for nid in est_ids}
    node_static = {}
    for nid in est_ids:
        I0, X0, A0 = nat[nid]
        ways = max(len(S(nid, CI)), 1) * max(len(S(nid, CX)), 1) * max(len(S(nid, CA)), 1)
        node_static[nid] = dict(
            I_native=int(I0), X_native=int(X0), A_native=int(A0),
            I_sup=int(supI[nid] and not I0), X_sup=int(supX[nid] and not X0), A_sup=int(supA[nid] and not A0),
            complete_conn=int(supX[nid] and supA[nid]), complete_chan=int(supA[nid]), eligible=eligible[nid],
            n_ways=int(ways if (supI[nid] and supX[nid] and supA[nid]) else 0),
            deliv_conn=deliv[nid]["conn"], deliv_chan=deliv[nid]["chan"], deliv_dir=deliv[nid]["dir"],
            plane_member=int(sum(nid in m_ for _, m_ in planes.values())),
            boundary_X=int(any(ext[idx[s]] for s in S(nid, CX) if s != nid)),
            boundary_A=int(any(ext[idx[s]] for s in S(nid, CA) if s != nid)),
            cut_I=int(len(S(nid, CI)) == 1 and nid not in S(nid, CI)),
            cut_X=int(len(S(nid, CX)) == 1 and nid not in S(nid, CX)),
            cut_A=int(len(S(nid, CA)) == 1 and nid not in S(nid, CA)),
            x_prob=float("nan"), a_prob=float("nan"), impact=float(spec.node(nid).impact))
    per_source: Dict[str, Dict[str, dict]] = {}
    reach_ungated = []
    est_arr = np.array(est)
    for s in sources:
        si = idx[s]
        tp_u = [v for v in adj_u[si] if not ext[v]]
        tp_g = [v for v in adj_g[si] if not ext[v]]
        dg = _bfs_multi(adj_g, tp_g, n_all) if tp_g else np.full(n_all, -1)
        du = _bfs_multi(adj_u, tp_u, n_all) if tp_u else np.full(n_all, -1)
        tp_m = [v for v in adj_m[si] if not ext[v]]
        dm = _bfs_multi(adj_m, tp_m, n_all) if tp_m else np.full(n_all, -1)
        reach_planes = {g for g, (ctl, mem) in planes.items() if ctl == s or dg[idx[ctl]] >= 0}
        doms = _dominators(adj_g, tp_g, n_all) if tp_g else [None] * n_all
        def cut_for(nid, cnd):
            """Paper 1B eq. (10): a supplying node that lies on every available supply of the
            condition. Direct single supplier, or a common dominator of every delivering supplier."""
            if nat[nid][{CI: 0, CX: 1, CA: 2}[cnd]]:
                return ""
            sups = {sid: rc for sid, rc in S(nid, cnd).items() if sid != nid}
            if not sups:
                return ""
            if len(sups) == 1:
                (sid, rc), = sups.items()
                return f"{sid}:{rc.value}"
            if cnd == CA and all(rc == RC.CONTROL_PLANE_CONFERRING for rc in sups.values()):
                return ""                    # standing conferral needs no carriage: no shared upstream
            common = None
            for sid in sups:
                dset = doms[idx[sid]] if idx[sid] < n_all else None
                if dset is None:
                    continue
                common = set(dset) if common is None else common & dset
            common = {ids[c] for c in common} - {nid} if common else set()
            return ";".join(sorted(f"{c}:upstream" for c in common))
        # primary H6 rival: ungated distance and redundancy on the subgraph induced by eligible nodes
        elig_set = {idx[nid] for nid in est_ids if eligible[nid] and du[idx[nid]] >= 0}
        adj_i = [[v for v in adj_u[u] if v in elig_set] if u in elig_set or u == si else [] for u in range(n_all)]
        tp_i = [v for v in tp_u if v in elig_set]
        di = _bfs_multi(adj_i, tp_i, n_all) if tp_i else np.full(n_all, -1)
        rows = {}
        for k in est:
            nid = ids[k]
            red = 1 if dg[k] == 0 else (_disjoint_paths(adj_g, tp_g, k, n_all, cap) if dg[k] > 0 else 0)
            redu = 1 if du[k] == 0 else (_disjoint_paths(adj_u, tp_u, k, n_all, cap) if du[k] > 0 else 0)
            redi = 1 if di[k] == 0 else (_disjoint_paths(adj_i, tp_i, k, n_all, cap) if di[k] > 0 else 0)
            hubs = [g for g in reach_planes if nid in planes[g][1]]
            cI, cX, cA = cut_for(nid, CI), cut_for(nid, CX), cut_for(nid, CA)
            rows[nid] = dict(gated_distance=int(dg[k]), ungated_distance=int(du[k]), induced_distance=int(di[k]),
                             redundancy=int(red), ungated_redundancy=int(redu), induced_redundancy=int(redi),
                             cut_I_src=cI, cut_X_src=cX, cut_A_src=cA,
                             cut_any=int(bool(cI or cX or cA)), mech_reachable=int(dm[k] >= 0),
                             hub_member=len(hubs), hub_max_fanout=max((len(planes[g][1]) for g in hubs), default=0),
                             touchpoint=int(k in tp_u))
        per_source[s] = rows
        reach_ungated.append(float((du[est_arr] >= 0).mean()))
    cut = {rc.value: 0 for rc in RC}
    for nid in est_ids:
        for cnd in Cond:
            s_ = S(nid, cnd)
            if len(s_) == 1:
                (sid, rc), = s_.items()
                if sid != nid:
                    cut[rc.value] += 1
    fan_max = {}
    for cls, key in ((RC.CONNECTION, "fanout_max_conn"), (RC.CHANNEL, "fanout_max_chan"),
                     (RC.CONTROL_PLANE_CONFERRING, "fanout_max_conf"), (RC.CONTROL_PLANE_DIRECTING, "fanout_max_dir")):
        fan_max[key] = max((len(v) for (s_, rc), v in fan.items() if rc == cls.value), default=0)
    supplied_corners = [(nid, cnd) for nid in est_ids for cnd in Cond
                        if len([s for s in S(nid, cnd) if s != nid]) > 0]
    red_mean = float(np.mean([len([s for s in S(nid, cnd) if s != nid]) for nid, cnd in supplied_corners])) if supplied_corners else 0.0
    completes_by_supplier: Dict[str, set] = {}
    for nid in est_ids:
        if supX[nid] and supA[nid]:
            for cnd in Cond:
                for s in S(nid, cnd):
                    if s != nid:
                        completes_by_supplier.setdefault(s, set()).add(nid)
    hub_conc = max((len(v) for v in completes_by_supplier.values()), default=0) / N
    deg_out = np.array([len(a_) for a_ in adj_u], dtype=float)
    est_mask = ~ext
    dmax = deg_out[est_mask].max() if est_mask.any() else 0
    centralisation = float((dmax - deg_out[est_mask]).sum() / max((N - 1) * (N - 2), 1)) if N > 2 else 0.0
    cut_adj = [[] for _ in ids]
    for nid in est_ids:
        for cnd in Cond:
            s_ = S(nid, cnd)
            if len(s_) == 1:
                (sid, _), = s_.items()
                if sid != nid:
                    cut_adj[idx[sid]].append(idx[nid])
    chain_depth = 0
    for s_ in range(n_all):
        if cut_adj[s_]:
            chain_depth = max(chain_depth, int(_bfs_multi(cut_adj, [s_], n_all).max()))
    betw = _betweenness_directed(adj_u, n_all)
    dists = [_bfs_multi(adj_u, [s_], n_all) for s_ in est]
    pl = [d_[d_ > 0].mean() for d_ in dists if (d_ > 0).any()]
    src_deg = float(np.mean([len([v for v in adj_u[idx[s]] if not ext[v]]) for s in sources]))
    arch = dict(
        n_estate=len(est), n_external=int(ext.sum()), source_degree=src_deg, n_relations=len(spec.relations),
        f_IXA=float(np.mean([supI[i] and supX[i] and supA[i] for i in est_ids])),
        f_XA=float(np.mean([supX[i] and supA[i] for i in est_ids])),
        f_A=float(np.mean([supA[i] for i in est_ids])),
        f_XA_native=float(np.mean([nat[i][1] and nat[i][2] for i in est_ids])),
        f_A_native=float(np.mean([nat[i][2] for i in est_ids])),
        complete_count=int(sum(supX[i] and supA[i] for i in est_ids)),
        redundancy_mean=red_mean, hub_concentration=float(hub_conc), centralisation=centralisation,
        chain_depth=int(chain_depth),
        cut_conn=cut[RC.CONNECTION.value] / N, cut_chan=cut[RC.CHANNEL.value] / N,
        cut_conf=cut[RC.CONTROL_PLANE_CONFERRING.value] / N, cut_dir=cut[RC.CONTROL_PLANE_DIRECTING.value] / N,
        **fan_max, boundary_share=ext_rows / n_rows if n_rows else 0.0,
        deg_mean=float(deg_out[est_mask].mean()) if est_mask.any() else 0.0, deg_max=int(dmax),
        betw_mean=float(betw[est_mask].mean()) if est_mask.any() else 0.0, betw_max=float(betw.max()),
        reach_ungated=float(np.mean(reach_ungated)) if reach_ungated else 0.0,
        n_scc=int(_scc_count(adj_u, n_all)), path_mean=float(np.mean(pl)) if pl else 0.0)
    return node_static, per_source, arch


# ---- runs (Section 8) --------------------------------------------------------

def _det_reach(spec, source: str) -> List[str]:
    c = _core()
    dspec = make_spec(spec.id, spec.nodes, spec.relations, spec.seed, deterministic=True)
    net = c.build_network(dspec, np.random.default_rng(1))
    r = c.run_one_trial(net, np.random.default_rng(1), entry=net.index[source])
    return sorted(n for n in r["reached_set"] if not net.external[net.index[n]])


def run_architecture(task: dict):
    c = _core()
    if task["family"] == "mixed":
        spec, meta, ameta = build_mixed(task)
    elif task["family"] == "representative":
        spec, meta, ameta = build_representative(task)
    else:
        spec, meta, ameta = build_canonical(task)
    sources = ameta["sources"].split(";")
    node_static, per_source, arch = measure(spec, sources)
    net = c.build_network(spec, np.random.default_rng(task["seed"]))
    est_ids = [nid for nid in net.ids if not net.external[net.index[nid]]]
    for nid in est_ids:
        node_static[nid]["x_prob"] = float(net.x_prob[net.index[nid]])
        node_static[nid]["a_prob"] = float(net.a_prob[net.index[nid]])
    for s, rows in per_source.items():
        for nid, r in rows.items():
            if r["ungated_distance"] < 0 and r["gated_distance"] >= 0:
                raise ICFailure(f"IC6 failed: gated reachable but ungated unreachable at {nid} in {task['arch_id']}")
            if r["gated_distance"] >= 0 and r["gated_distance"] < r["ungated_distance"]:
                raise ICFailure(f"IC6 failed: gated distance below ungated at {nid} in {task['arch_id']}")
            if r["redundancy"] > r["ungated_redundancy"]:
                raise ICFailure(f"IC6 failed: gated redundancy above ungated at {nid} in {task['arch_id']}")
            if r["gated_distance"] >= 0 and r["redundancy"] < 1:
                raise ICFailure(f"IC6 failed: reachable node with redundancy 0 at {nid} in {task['arch_id']}")
    if task["family"] in ("canonical", "representative") and ameta.get("n_A_assigned", -1) != int(round(task["f"] * task["N"])):
        raise ICFailure(f"IC4 failed: A count {ameta.get('n_A_assigned')} for f {task['f']}, N {task['N']} in {task['arch_id']}")
    n_runs = task["n_runs"] * len(sources)          # Section 8: the configured runs for each source
    per_src = [task["n_runs"]] * len(sources)
    events = {s: {nid: 0 for nid in est_ids} for s in sources}
    runs, fr_all, coll, imp_all, src_of = [], [], [], [], []
    hit_any = 0
    elig_union = {s: {nid for nid in est_ids if node_static[nid]["eligible"] and per_source[s][nid]["mech_reachable"]}
                  for s in sources}
    t = 0
    for s, ns in zip(sources, per_src):
        for _ in range(ns):
            rs = task["seed"] * 100_003 + t
            t += 1
            r = c.run_one_trial(net, np.random.default_rng(rs), entry=net.index[s])
            first = {}
            for s_, n_, _ in r["reached"]:
                if n_ in events[s]:
                    first.setdefault(n_, s_)
            for n_ in first:
                events[s][n_] += 1
                if n_ not in elig_union[s]:
                    raise ICFailure(f"IC2 failed: {n_} compromised outside the eligible union of {s} in {task['arch_id']}")
            fr = len(first) / len(est_ids)
            fr_all.append(fr); coll.append(bool(r["collapsed"])); hit_any += len(first) > 0; src_of.append(s)
            imp_all.append(float(sum(node_static[n_]["impact"] for n_ in first)))
            runs.append(dict(arch_id=task["arch_id"], source=s, run_seed=rs, n_compromised=len(first),
                             fraction=fr, collapsed=bool(r["collapsed"]), n_steps=r["n_steps"],
                             first_compromised=";".join(f"{n}:{k}" for n, k in first.items())))
    fr_all = np.array(fr_all); coll = np.array(coll); imp_all = np.array(imp_all); src_of = np.array(src_of)
    for s in sources:
        if fr_all[src_of == s].max() > len(elig_union[s]) / len(est_ids) + 1e-9:
            raise ICFailure(f"IC2 failed: Y exceeds the eligible union for {s} in {task['arch_id']}")
    if task["family"] != "mixed" and fr_all.max() > arch["f_A"] + 1e-9:
        raise ICFailure(f"IC2 failed: Y exceeds f_A in {task['arch_id']}")
    nrows = []
    for s, ns in zip(sources, per_src):
        for nid in est_ids:
            nrows.append(dict(arch_id=task["arch_id"], source=s, node=nid, role=meta.get(nid, ""),
                              events=events[s][nid], trials=ns, p=events[s][nid] / max(ns, 1),
                              **node_static[nid], **per_source[s][nid]))
    outcomes, src_means = [], []
    for s in sources:
        m = src_of == s
        y = fr_all[m]
        src_means.append(float(y.mean()))
        outcomes.append(dict(arch_id=task["arch_id"], source=s, Y=float(y.mean()), var_Y=float(y.var()),
                             se_Y=float(y.std(ddof=1) / np.sqrt(len(y))) if len(y) > 1 else float("nan"),
                             first_hop_rate=float((y > 0).mean()), P_collapse=float(coll[m].mean()),
                             impact_reached=float(imp_all[m].mean()), n_runs=int(m.sum()), var_share_source=float("nan")))
    ss_b = sum((src_of == s).sum() * (mu - fr_all.mean()) ** 2 for s, mu in zip(sources, src_means))
    ss_t = float(((fr_all - fr_all.mean()) ** 2).sum())
    outcomes.append(dict(arch_id=task["arch_id"], source="pooled", Y=float(np.mean(src_means)), var_Y=float(fr_all.var()),
                         se_Y=float(np.nanmean([o["se_Y"] for o in outcomes]) / np.sqrt(len(sources))),
                         first_hop_rate=hit_any / n_runs, P_collapse=float(coll.mean()),
                         impact_reached=float(imp_all.mean()), n_runs=n_runs,
                         var_share_source=ss_b / ss_t if ss_t > 0 else float("nan")))
    arow = dict(task["meta"], arch_id=task["arch_id"], family=task["family"], shape=task.get("shape", "mixed"),
                cell=task.get("cell", ""), seed=task["seed"], N=arch["n_estate"],
                **ameta,
                **{f"f_{k}": v for k, v in task.get("factors", {}).items()}, **arch)
    ic3 = None
    if task.get("ic3"):
        ic3 = dict(arch_id=task["arch_id"], shape=task["shape"], variant=task.get("variant", ""),
                   reached=";".join(_det_reach(spec, sources[0])))
    return arow, outcomes, nrows, runs, ic3, (spec if task.get("want_spec") else None)


# ---- tasks and generation ----------------------------------------------------

def canonical_tasks(sample: str, seed: int, N: int, per_cell: int, n_runs: int) -> List[dict]:
    T = []
    fs = DESIGN["f_levels"]
    def mk(shape, cell, k, **kw):
        h = hashlib.sha256(f"{seed}|{shape}|{cell}|{k}".encode()).digest()
        aseed = int.from_bytes(h[:4], "little") % (2 ** 31 - 1) + 1
        aid = f"{sample}_{seed}_{shape}_{cell}_{k:03d}" + (("_" + kw["variant"]) if kw.get("variant") else "")
        t = dict(arch_id=aid, family="canonical", shape=shape, cell=cell, seed=aseed, N=N, n_runs=n_runs,
                 meta=dict(sample=sample, sample_seed=seed, block="", is_primary=True), **kw)
        return t
    for f in fs:
        for j, d in enumerate(mesh_densities(N, f)):
            for k in range(per_cell):
                T.append(mk("mesh", f"f{f}_j{j:02d}", k, f=f, d=d, ic3=(f == 1.0 and k == 0)))
        for n60 in DESIGN["star_n"]:
            n = scale60(N, n60)
            for shape in ("star_cp", "star_conn"):
                for k in range(per_cell):
                    variants = ["intact", "hub_removed"] + [f"leaves_removed_{scale60(N, r)}" for r in DESIGN["star_leaf_removals"]]
                    for v in variants:
                        t = mk(shape, f"f{f}_n{n}", k, f=f, n=n, variant=v, ic3=(f == 1.0 and k == 0 and v in ("intact", "hub_removed")))
                        t["meta"]["block"] = f"{sample}_{seed}_{shape}_f{f}_n{n}_{k:03d}"
                        t["meta"]["is_primary"] = v == "intact"
                        T.append(t)
            for k in range(per_cell):
                t = mk("star_hubsource", f"f{f}_n{n}", k, f=f, n=n, variant="hubsource", ic3=(f == 1.0 and k == 0))
                t["meta"]["is_primary"] = False
                T.append(t)
        for shape in ("chain_chan", "chain_conn"):
            for k in range(per_cell):
                T.append(mk(shape, f"f{f}", k, f=f, ic3=(f == 1.0 and k == 0)))
    return T


def matched_tasks(sample: str, seed: int, N: int, blocks: int, n_runs: int) -> List[dict]:
    T = []
    for f in DESIGN["f_levels"]:
        for b in range(blocks):
            h = hashlib.sha256(f"{seed}|matched|{f}|{b}".encode()).digest()
            aseed = int.from_bytes(h[:4], "little") % (2 ** 31 - 1) + 1
            block = f"{sample}_{seed}_matched_f{f}_{b:03d}"
            for shape in ("matched_mesh", "matched_star", "matched_chain"):
                T.append(dict(arch_id=f"{block}_{shape.split('_')[1]}", family="matched", shape=shape,
                              cell=f"f{f}", seed=aseed, N=N, n_runs=n_runs, f=f,
                              meta=dict(sample=sample, sample_seed=seed, block=block, is_primary=True)))
    return T


def representative_tasks(sample: str, seed: int, N: int, blocks: int, n_runs: int) -> List[dict]:
    T = []
    for f in DESIGN["f_levels"]:
        for e in DESIGN["entry_shares"]:
            for b in range(blocks):
                h = hashlib.sha256(f"{seed}|representative|{f}|{e}|{b}".encode()).digest()
                aseed = int.from_bytes(h[:4], "little") % (2 ** 31 - 1) + 1
                block = f"{sample}_{seed}_rep_f{f}_e{e}_{b:03d}"
                shapes = [f"rep_mesh{d}" for d in DESIGN["mesh_degrees"]] + ["rep_star", "rep_chain"]
                for shape in shapes:
                    T.append(dict(arch_id=f"{block}_{shape[4:]}", family="representative", shape=shape,
                                  cell=f"f{f}_e{e}", seed=aseed, N=N, n_runs=n_runs, f=f, entry_share=e,
                                  meta=dict(sample=sample, sample_seed=seed, block=block, is_primary=True)))
    return T


def mixed_tasks(sample: str, seed: int, ranges: dict, n: int, n_runs: int) -> List[dict]:
    T = []
    for k, f in enumerate(draw_factors(n, ranges, seed * 100 + 7)):
        h = hashlib.sha256(f"{seed}|mixed|{k}".encode()).digest()
        aseed = int.from_bytes(h[:4], "little") % (2 ** 31 - 1) + 1
        T.append(dict(arch_id=f"{sample}_{seed}_mixed_{k:04d}", family="mixed", shape="mixed", cell="", seed=aseed,
                      index=k, n_runs=n_runs, factors=f, meta=dict(sample=sample, sample_seed=seed, block="", is_primary=True)))
    return T


def all_tasks(design: dict, pilot: bool) -> List[dict]:
    seeds = design["pilot_seeds"] if pilot else design["seeds"]
    per_cell = design["pilot_arch_per_cell"] if pilot else design["arch_per_cell"]
    blocks = design["pilot_matched_blocks"] if pilot else design["matched_blocks"]
    n_mixed = design["pilot_mixed_per_sample"] if pilot else design["mixed_per_sample"]
    n_runs = design["pilot_runs_per_arch"] if pilot else design["runs_per_arch"]
    plan = [("development", seeds["development"], FACTORS_DEV, design["n_canonical"])]
    plan += [("replication", s, FACTORS_DEV, design["n_canonical"]) for s in seeds["replication"]]
    plan += [("heldout", seeds["heldout"], FACTORS_HELDOUT, design["n_canonical_heldout"])]
    T = []
    for sample, seed, ranges, N in plan:
        T += canonical_tasks(sample, seed, N, per_cell, n_runs)
        T += matched_tasks(sample, seed, N, blocks, n_runs)
        T += representative_tasks(sample, seed, N, design["pilot_rep_blocks"] if pilot else design["rep_blocks"], n_runs)
        T += mixed_tasks(sample, seed, ranges, n_mixed, n_runs)
    return T


ARCH_COLUMNS = (["arch_id", "sample", "sample_seed", "block", "is_primary", "family", "shape", "cell", "seed", "N",
                 "f_target", "d", "n", "variant", "sources", "source_types", "N_target", "n_A_assigned", "entry_share"]
                + [f"f_{k}" for k in FACTORS_DEV] + ["n_estate", "n_external", "n_relations"] + ARCH_MEASURES)


def _append(path: str, rows: List[dict], columns: Optional[List[str]] = None) -> None:
    """Append rows with a fixed column set so every family writes the same header."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    if columns is not None:
        df = df.reindex(columns=columns)
    if path.endswith(".gz"):
        buf = io.StringIO()
        df.to_csv(buf, index=False, header=not os.path.exists(path))
        with gzip.open(path, "at") as f:
            f.write(buf.getvalue())
    else:
        df.to_csv(path, mode="a", index=False, header=not os.path.exists(path))


def generate(run_dir: str, design: dict, workers: int, pilot: bool, cfg: dict) -> None:
    tasks = all_tasks(design, pilot)
    for t in tasks:
        t["want_spec"] = cfg.get("write_specs", True)
    paths = {k: os.path.join(run_dir, f) for k, f in dict(
        arch="architectures.csv", out="outcomes.csv", nodes="nodes.csv.gz", runs="runs.csv.gz", ic3="ic3_deterministic.csv").items()}
    if cfg.get("write_specs", True):
        os.makedirs(os.path.join(run_dir, "specs"), exist_ok=True)
    marker = os.path.join(run_dir, "completed.txt")
    done = set(open(marker).read().split()) if os.path.exists(marker) else set()
    todo = [t for t in tasks if t["arch_id"] not in done]
    print(f"  {len(tasks)} architectures x {tasks[0]['n_runs']} runs; {len(done)} done, {len(todo)} to run, {workers} worker(s)")
    t0, n = time.time(), 0

    def sink(res):
        nonlocal n
        arow, outs, nrows, runs, ic3, spec = res
        _append(paths["arch"], [arow], ARCH_COLUMNS); _append(paths["out"], outs); _append(paths["nodes"], nrows)
        if cfg.get("write_runs", True):
            _append(paths["runs"], runs)
        if ic3:
            _append(paths["ic3"], [ic3])
        if spec is not None:
            with open(os.path.join(run_dir, "specs", f"{arow['arch_id']}.yaml"), "w") as fh:
                fh.write(spec_to_yaml(spec))
        with open(marker, "a") as fh:
            fh.write(arow["arch_id"] + "\n")     # written last: an architecture counts as done only now
        n += 1
        if n in (1, 5, 20) or n % max(1, len(todo) // 20) == 0 or n == len(todo):
            el = time.time() - t0
            print(f"    {n}/{len(todo)}  elapsed {el/60:.1f} min  eta {el/n*(len(todo)-n)/60:.1f} min", flush=True)

    if workers <= 1:
        for t in todo:
            try:
                sink(run_architecture(t))
            except ICFailure as e:
                raise SystemExit(f"  {e}\n  Generation stopped: an implementation check failed.")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_architecture, t) for t in todo]
            for fut in as_completed(futs):
                try:
                    res = fut.result()
                except ICFailure as e:
                    for f_ in futs:
                        f_.cancel()
                    raise SystemExit(f"  {e}\n  Generation stopped: an implementation check failed.")
                sink(res)


def time_budget(design: dict, workers: int) -> dict:
    tasks = [t for t in all_tasks(design, False) if t["meta"]["sample"] == "development"]
    rng = np.random.default_rng(0)
    pick = [tasks[i] for i in rng.choice(len(tasks), 50, replace=False)]
    for t in pick:
        t["want_spec"] = False
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(run_architecture, pick))
    wall = time.time() - t0
    total = len(all_tasks(design, False))
    est_h = wall / len(pick) * total / 3600
    print(f"  50 architectures at {design['runs_per_arch']} runs: {wall:.0f} s wall on {workers} workers")
    print(f"  full design, {total} architectures: about {est_h:.1f} h")
    return dict(sampled=50, wall_s=wall, workers=workers, total_architectures=total, estimate_h=est_h)


# ---- scenario file emitter and run record --------------------------------------

def spec_to_yaml(spec) -> str:
    L = [f"id: {spec.id}", f"spec_version: '{spec.spec_version}'", f"description: {spec.description}", "governance:"]
    for g in spec.governance:
        L += [f"- id: {g.id}", f"  defender_controlled: {str(g.defender_controlled).lower()}"]
    L.append("nodes:")
    for n in spec.nodes:
        L += [f"- id: {n.id}", f"  governance: {n.governance}", f"  interface: {str(n.interface).lower()}",
              f"  execution_pathway: {str(n.execution_pathway).lower()}", f"  authority: {str(n.authority).lower()}",
              f"  state_object: {n.state_object}", f"  visible: {str(n.visible).lower()}", f"  impact: {n.impact:.6g}"]
    L.append("relations:")
    for r in spec.relations:
        L += [f"- supplier: {r.supplier}", f"  receiver: {r.receiver}", f"  condition: {r.condition.value}",
              f"  relation_class: {r.relation_class.value}", f"  conferral: {str(r.conferral).lower()}"]
        if r.group:
            L.append(f"  group: {r.group}")
    r_ = spec.rates
    L += [f"entry_certain: {str(spec.entry_certain).lower()}", f"seed: {spec.seed}",
          f"deterministic: {str(spec.deterministic).lower()}",
          "ablation:", f"  structure: {str(spec.ablation.structure).lower()}",
          f"  time: {str(spec.ablation.time).lower()}", f"  adaptation: {str(spec.ablation.adaptation).lower()}",
          "rates:"] + [f"  {k}: {getattr(r_, k)}" for k in ("threat_capability", "authority_rate", "execution_rate",
                                                             "control_variance", "beta_conn", "synchrony", "dependency_factor",
                                                             "execution_drift_boost", "control_plane_takeover_rate", "authority_boost",
                                                             "phantom_cp_rate", "phantom_channel_rate", "phantom_dep_rate")]
    L += ["time:", f"  max_steps: {spec.time.max_steps}", f"  latency_steps: {spec.time.latency_steps}",
          f"  drift_increment: {spec.time.drift_increment}", f"  exfil_dwell_steps: {spec.time.exfil_dwell_steps}",
          "adaptation:", f"  remediation_capability: {spec.adaptation.remediation_capability}",
          f"  threshold: {spec.adaptation.threshold}", f"  simple_policy: {str(spec.adaptation.simple_policy).lower()}"]
    return "\n".join(L) + "\n"


def code_hash() -> str:
    src = open(os.path.abspath(__file__), "rb").read().replace(b"\r\n", b"\n").decode("utf-8")
    src = re.sub(r"# >>> RUN_CONFIG.*?# <<< RUN_CONFIG", "", src, flags=re.S)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def run_record(design: dict, fz: dict, pilot: bool) -> dict:
    return json.loads(json.dumps(dict(
        code_version=CODE_VERSION, spec_version=SPEC_VERSION, code_sha256=code_hash(),
        core_version=fz["core_version"], core_digest=fz["freeze_digest"], hash_check=fz["note"],
        pilot=pilot, design=design, factors_dev=FACTORS_DEV, factors_heldout=FACTORS_HELDOUT,
        fixed=FIXED, source_types=SOURCE_TYPES, gated=GATED, ungated=UNGATED,
        python=sys.version.split()[0], rng="numpy PCG64 (default_rng)",
        packages={m: getattr(__import__(m), "__version__", "n/a") for m in ("numpy", "pandas", "scipy")},
    ), default=str))


def write_or_check_record(run_dir: str, rec: dict) -> Tuple[bool, List[str]]:
    path = os.path.join(run_dir, "run_record.json")
    if not os.path.exists(path):
        rec = dict(rec, started=_dt.datetime.now().isoformat(timespec="seconds"))
        with open(path, "w") as f:
            json.dump(rec, f, indent=2)
        return True, []
    saved = json.load(open(path))
    diffs = [k for k in rec if saved.get(k) != rec[k]]
    return not diffs, diffs


# ############################################################################
# PART 2: ANALYSIS (Sections 9 to 15)
# ############################################################################

# ---- estimators in numpy (no sklearn) -----------------------------------------

class Standardiser:
    def __init__(self, X):
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd < 1e-12] = 1.0

    def __call__(self, X):
        return (X - self.mu) / self.sd


def binomial_ridge(X, k, n, lam=1.0, iters=60):
    """Binomial logistic regression (events k of trials n) with ridge on slopes, IRLS."""
    Xb = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(Xb.shape[1])
    P = lam * np.eye(Xb.shape[1]); P[0, 0] = 0
    for _ in range(iters):
        eta = Xb @ beta
        p = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
        W = n * p * (1 - p) + 1e-9
        g = Xb.T @ (k - n * p) - P @ beta
        H = Xb.T @ (Xb * W[:, None]) + P
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def binomial_firth(X, k, n, lam=0.0, iters=80):
    """Binomial logistic regression by Firth penalised likelihood (Heinze and Schemper), with an
    optional ridge on the slopes. Modified score: X'(k - n p + h (1/2 - p)) - lam beta_slopes."""
    Xb = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(Xb.shape[1])
    P = lam * np.eye(Xb.shape[1]); P[0, 0] = 0
    for _ in range(iters):
        eta = np.clip(Xb @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = n * p * (1 - p) + 1e-12
        XtW = Xb.T * W
        H = XtW @ Xb + P
        Hinv = np.linalg.pinv(H)
        h = np.einsum("ij,jk,ik->i", Xb, Hinv, Xb) * W          # leverages of the weighted fit
        g = Xb.T @ (k - n * p + h * (0.5 - p)) - P @ beta
        step = Hinv @ g
        if np.abs(step).max() > 5:
            step = step * (5 / np.abs(step).max())
        beta = beta + step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def fit_node_model(df: pd.DataFrame, cols: List[str], lam=0.0, sc: Optional[Standardiser] = None) -> dict:
    X = df[cols].values.astype(float)
    sc = sc or Standardiser(X)
    beta = binomial_firth(sc(X), df["events"].values.astype(float), df["trials"].values.astype(float), lam)
    return dict(cols=cols, sc=sc, beta=beta, lam=lam)


def cv_ridge(dev: pd.DataFrame, cols: List[str], grid: List[float], folds: int, seed: int) -> Tuple[float, dict]:
    """Section 13: any stability ridge is chosen by 5-fold cross-validation on development (by
    architecture) and frozen; the curve is recorded."""
    archs = dev["arch_id"].unique()
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(archs, rng.integers(0, folds, len(archs))))
    fid = dev["arch_id"].map(fold_of).values
    sc = Standardiser(dev[cols].values.astype(float))
    curve = {}
    for lam in grid:
        dev_ = 0.0
        for f in range(folds):
            tr, te = dev[fid != f], dev[fid == f]
            if len(tr) == 0 or len(te) == 0:
                continue
            m = fit_node_model(tr, cols, lam, sc)
            eta = np.clip(predict_node_model(m, te), -30, 30)
            p = 1 / (1 + np.exp(-eta))
            k, n = te["events"].values, te["trials"].values
            dev_ += -2 * float(np.sum(k * np.log(p + 1e-12) + (n - k) * np.log(1 - p + 1e-12)))
        curve[lam] = dev_
    best = min(curve, key=curve.get)
    return best, curve


def predict_node_model(m: dict, df: pd.DataFrame) -> np.ndarray:
    X = m["sc"](df[m["cols"]].values.astype(float))
    return np.column_stack([np.ones(len(X)), X]) @ m["beta"]


def concordance(s: np.ndarray, y: np.ndarray) -> float:
    """Share of pairs with different y whose score order matches the y order, score ties one half.
    Exact, O(n log n): sweep in increasing y with a Fenwick tree over compressed score ranks, so it
    is safe on hundreds of thousands of rows (the pilot broad screen)."""
    s, y = np.asarray(s, float), np.asarray(y, float)
    ok = np.isfinite(s) & np.isfinite(y)
    s, y = s[ok], y[ok]
    n = len(y)
    if n < 2:
        return float("nan")
    ranks = np.unique(s, return_inverse=True)[1] + 1          # 1-based compressed score ranks
    m = int(ranks.max())
    order = np.argsort(y, kind="mergesort")
    tree = np.zeros(m + 1, dtype=np.int64)
    def add(i):
        while i <= m:
            tree[i] += 1
            i += i & -i
    def prefix(i):
        t = 0
        while i > 0:
            t += tree[i]
            i -= i & -i
        return t
    conc = tied = total = 0.0
    seen = 0
    i = 0
    while i < n:
        j = i
        while j < n and y[order[j]] == y[order[i]]:
            j += 1
        grp = ranks[order[i:j]]
        for r in grp:                     # pairs with earlier (lower-y) rows only
            below = prefix(r - 1)
            same = prefix(r) - below
            conc += below
            tied += same
        total += seen * (j - i)
        for r in grp:
            add(r)
        seen += j - i
        i = j
    if total == 0:
        return float("nan")
    return float((conc + 0.5 * tied) / total)


def within_arch_concordance(df: pd.DataFrame, score: np.ndarray) -> pd.Series:
    """Concordance within each (architecture, source) unit, averaged to the architecture, so
    nodes under different sources are never compared with one another."""
    out = {}
    for (aid, src), idx in df.groupby(["arch_id", "source"]).indices.items():
        c = concordance(score[idx], df["p"].values[idx])
        if np.isfinite(c):
            out.setdefault(aid, []).append(c)
    return pd.Series({k: float(np.mean(v)) for k, v in out.items()})


def ci(v, alpha=0.05):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2))))


def boot_over(groups: List[np.ndarray], fn, n_boot: int, seed: int):
    """Bootstrap resampling within each group of indices; fn(list of index arrays)."""
    rng = np.random.default_rng(seed)
    return np.array([fn([rng.choice(g, len(g)) for g in groups]) for _ in range(n_boot)])


# ---- predictions (Section 9) ----------------------------------------------------

def per_step_probs(nodes: pd.DataFrame, shape: str) -> np.ndarray:
    T, bc, delta = FIXED["threat_capability"], FIXED["beta_conn"], FIXED["dependency_factor"]
    x, a = nodes["x_prob"].values, nodes["a_prob"].values
    if shape.endswith("chan"):
        return np.minimum(1.0, (1 - 1 / (1 + delta)) * a * T)
    return bc * x * a * T


def chain_prediction(nodes: pd.DataFrame, shape: str, H: int) -> np.ndarray:
    """p_k by dynamic programming over steps: first-passage distribution along the chain."""
    nodes = nodes.sort_values("depth")
    pi = per_step_probs(nodes, shape)
    complete = nodes["complete_chan"].values if shape.endswith("chan") else nodes["complete_conn"].values
    # first passage of node j at step t given node j-1 first compromised at step s: geometric from s+1
    prev = np.zeros(H + 1); prev[0] = 1.0            # source compromised at step 0
    out = []
    for j in range(len(nodes)):
        if not complete[j]:
            out += [0.0] * (len(nodes) - j)
            break
        cur = np.zeros(H + 1)
        for s in range(H):
            if prev[s] == 0:
                continue
            for t in range(s + 1, H + 1):
                cur[t] += prev[s] * (1 - pi[j]) ** (t - s - 1) * pi[j]
        out.append(float(cur.sum()))
        prev = cur
    return np.array(out)


def mesh_prediction(f_xa: float, N: int, d: float) -> Tuple[float, float]:
    """(threshold density d*, predicted Y) under the Section 9.4 approximation."""
    pi = FIXED["beta_conn"] * FIXED["execution_rate"] * FIXED["authority_rate"] * FIXED["threat_capability"]
    T_H = 1 - (1 - pi) ** FIXED["max_steps"]
    d_star = 1 / max((N - 1) * f_xa * T_H, 1e-9)
    c = d * (N - 1) * f_xa * T_H
    S = 0.0
    if c > 1:
        S = 1.0
        for _ in range(200):
            S = 1 - np.exp(-c * S)
    return d_star, f_xa * S * S


# ---- H1 mesh -------------------------------------------------------------------

def _logistic4(x, L, k, m):
    return L / (1 + np.exp(-k * (x - m)))


def fit_sigmoid(logd: np.ndarray, y: np.ndarray, L0: float) -> Tuple[float, float, float]:
    try:
        popt, _ = optimize.curve_fit(_logistic4, logd, y, p0=[max(L0, 0.05), 3.0, float(np.median(logd))],
                                     bounds=([0, 0, logd.min() - 5], [1.0, 100, logd.max() + 5]), maxfev=4000)
        return float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:
        return float("nan"), float("nan"), float("nan")


def h1_mesh(df: pd.DataFrame, design: dict) -> dict:
    d = df[df["shape"] == "mesh"].copy()
    d["logd"] = np.log(d["d"])
    fs = sorted(d["f_target"].unique())
    cells = {(f, dd): g.index.values for (f, dd), g in d.groupby(["f_target", "d"])}
    def fit_all(idx_lists):
        rows = []
        cell_keys = list(cells)
        means = {k: d.loc[ix, "Y"].mean() for k, ix in zip(cell_keys, idx_lists)}
        res = {}
        for f in fs:
            ks = [k for k in cell_keys if k[0] == f]
            x = np.array([np.log(k[1]) for k in ks]); y = np.array([means[k] for k in ks])
            f_xa = d.loc[[cells[k][0] for k in ks], "f_XA"].mean()
            L, kk, m = fit_sigmoid(x, y, f_xa)
            res[f] = dict(L=L, slope=kk, midpoint=float(np.exp(m)) if np.isfinite(m) else float("nan"),
                          y_min=float(y.min()), y_max=float(y.max()), f_xa=float(f_xa),
                          plateau_gap=float(abs(y[np.argmax(x)] - f_xa)))
        return res
    point = fit_all([cells[k] for k in cells])
    boots = boot_over([cells[k] for k in cells], fit_all, design["n_boot"], design["boot_seed"])
    out = {}
    for f in fs:
        r = point[f]
        slopes = np.array([b[f]["slope"] for b in boots]); mids = np.array([b[f]["midpoint"] for b in boots])
        out[f] = dict(**r, slope_ci=ci(slopes), midpoint_ci=ci(mids),
                      rise=bool(r["y_min"] < 0.10 * r["f_xa"] and r["y_max"] > 0.50 * r["f_xa"]),
                      d_star_pred=mesh_prediction(r["f_xa"], int(d["N"].iloc[0]), 0.01)[0])
    mids = np.array([point[f]["midpoint"] for f in fs]); fx = np.array([point[f]["f_xa"] for f in fs])
    rho = float(stats.spearmanr(fx, mids).correlation)
    rho_b = np.array([stats.spearmanr(fx, [b[f]["midpoint"] for f in fs]).correlation for b in boots])
    prod = mids * fx
    cv = float(np.nanstd(prod) / np.nanmean(prod)) if np.nanmean(prod) > 0 else float("nan")
    crit = dict(a_rise_and_slope=bool(all(out[f]["rise"] and out[f]["slope_ci"][0] > 0 for f in fs)),
                b_midpoint_falls=bool(rho < 0 and ci(rho_b)[1] < 0),
                c_scaling_cv=bool(cv < design["h1_cv_max"]),
                d_plateau=bool(all(out[f]["plateau_gap"] <= design["h1_plateau_tol"] for f in fs)))
    return dict(per_f=out, spearman_f_midpoint=rho, spearman_ci=ci(rho_b), midpoint_x_f=prod.tolist(), cv=cv,
                criteria=crit, passed=bool(all(crit.values())))


# ---- H2 star -------------------------------------------------------------------

def h2_star(df: pd.DataFrame, design: dict) -> dict:
    d = df[df["shape"].isin(["star_cp", "star_conn"])].copy()
    k_large = {N: scale60(int(N), max(design["star_leaf_removals"])) for N in d["N_target"].unique()}
    d["variant2"] = d.apply(lambda r: "leaves_removed_k" if r["variant"] == f"leaves_removed_{k_large[r['N_target']]}" else r["variant"], axis=1)
    d = d[d["variant2"].isin(["intact", "hub_removed", "leaves_removed_k"])]
    piv = d.pivot_table(index=["shape", "f_target", "n", "block"], columns="variant2", values="Y").reset_index()
    piv["dY_hub"] = piv["intact"] - piv["hub_removed"]
    piv["dY_leaves"] = piv["intact"] - piv["leaves_removed_k"]
    piv["contrast"] = piv["dY_hub"] - piv["dY_leaves"]
    out, ok = {}, True
    for (shape, f, n), g in piv.groupby(["shape", "f_target", "n"]):
        v = g["contrast"].dropna().values
        b = boot_over([np.arange(len(v))], lambda ix: v[ix[0]].mean(), design["n_boot"], design["boot_seed"])
        lo, hi = ci(b)
        out[f"{shape}|f{f}|n{n}"] = dict(n_bases=int(len(v)), mean_contrast=float(v.mean()), ci=(lo, hi),
                                         dY_hub=float(g["dY_hub"].mean()), dY_leaves=float(g["dY_leaves"].mean()))
        ok = ok and lo > 0
    return dict(cells=out, passed=bool(ok))


# ---- H3 chain ------------------------------------------------------------------

def _h3_cell(args):
    """One (form, f) cell: observed profile, prediction, simultaneous band (Section 13)."""
    shape, f, chains, H, n_sim, seed, sp_max, f1_tol = args
    rng = np.random.default_rng(seed)
    depth = max(len(c_["pi"]) for c_ in chains)
    obs = np.zeros(depth); pred = np.zeros(depth); ntot = 0
    for c_ in chains:
        obs += c_["obs"] * c_["trials"]; pred += c_["pred"] * c_["trials"]; ntot += c_["trials"]
    obs /= ntot; pred /= ntot
    var = pred * (1 - pred) / ntot
    zero = var <= 1e-15                    # depths with zero predicted variance: unstandardised comparison
    se = np.sqrt(np.where(zero, 1.0, var))
    maxdev, maxstep = np.empty(n_sim), np.empty(n_sim)
    for b in range(n_sim):
        acc = np.zeros(depth)
        for c_ in chains:
            W = rng.geometric(np.clip(c_["pi"], 1e-9, 1.0), size=(c_["trials"], depth))
            reach = (np.cumsum(W, axis=1) <= H) & np.cumprod(c_["complete"])[None, :].astype(bool)
            acc += reach.sum(0)
        sim = acc / ntot
        maxdev[b] = np.max(np.abs(sim - pred) / se)
        maxstep[b] = np.max(np.diff(sim)) if depth > 1 else 0.0
    level = 100.0 * (1.0 - 0.05 / DESIGN["h3_family_cells"])      # family-wise 5% per sample
    crit = float(np.percentile(maxdev, level))
    crit_step = float(np.percentile(maxstep, level))
    dev_vec = np.abs(obs - pred) / se
    obs_dev = float(np.max(dev_vec))
    inside = bool(obs_dev <= crit and np.all(np.abs(obs - pred)[zero] < 1e-12))
    obs_step = float(np.max(np.diff(obs))) if depth > 1 else 0.0
    mono = bool(obs_step <= crit_step + 1e-12)
    f1_ok = bool(np.max(np.abs(obs - pred)) <= f1_tol) if f == 1.0 else True
    cell_ok = bool(mono and inside and f1_ok)
    degenerate = bool(np.all(pred == 0) and np.all(obs == 0))
    return f"{shape}|f{f}", dict(max_positive_step=obs_step, step_critical=crit_step, monotone=mono, degenerate_all_zero=degenerate,
                                 inside_band=inside, band_critical=crit, observed_maxdev=obs_dev,
                                 f1_within_tol=f1_ok, passed=cell_ok, observed=obs.round(4).tolist(),
                                 predicted=pred.round(4).tolist(), max_abs_gap=float(np.abs(obs - pred).max()))


def h3_chain(arch: pd.DataFrame, nodes: pd.DataFrame, design: dict, workers: int = 1) -> Tuple[dict, pd.DataFrame]:
    H = FIXED["max_steps"]
    chains = arch[arch["shape"].isin(["chain_chan", "chain_conn"])]
    nn = nodes[nodes["arch_id"].isin(chains["arch_id"])].copy()
    nn = nn[nn["role"].astype(str).str.startswith("depth")].copy()
    nn["depth"] = nn["role"].str[5:].astype(int)
    jobs, pred_rows = [], []
    for (shape, f), g in chains.groupby(["shape", "f_target"]):
        cl = []
        for aid in g["arch_id"]:
            cn = nn[nn["arch_id"] == aid].sort_values("depth")
            pk = chain_prediction(cn, shape, H)
            complete = (cn["complete_chan"].values if shape.endswith("chan") else cn["complete_conn"].values).astype(int)
            cl.append(dict(pi=per_step_probs(cn, shape), complete=complete, obs=cn["p"].values.astype(float),
                           pred=pk, trials=int(cn["trials"].iloc[0])))
            for k, (p, pr) in enumerate(zip(cn["p"].values, pk), 1):
                pred_rows.append(dict(arch_id=aid, node=cn["node"].values[k - 1], depth=k, observed=p, predicted=pr))
        jobs.append((shape, f, cl, H, design["n_boot"], design["boot_seed"], None, design["h3_f1_tol"]))
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
            results = list(ex.map(_h3_cell, jobs))
    else:
        results = [_h3_cell(j) for j in jobs]
    out = dict(results)
    return dict(cells=out, passed=bool(all(c["passed"] for c in out.values()))), pd.DataFrame(pred_rows)


# ---- H4 position and H6 gate ------------------------------------------------------

H4_COLS = ["gated_distance", "log_redundancy", "dist_x_red", "gated_unreachable"]


def _node_frame(nodes: pd.DataFrame, arch: pd.DataFrame) -> pd.DataFrame:
    """Mixed-architecture node rows with the H4 and H6 encodings (Section 13). Directing-plane
    edges already enter distance and redundancy; membership is not a separate term."""
    m = nodes.merge(arch[["arch_id", "sample", "family"]], on="arch_id")
    m = m[m["family"] == "mixed"].copy()
    m["gated_unreachable"] = (m["gated_distance"] < 0).astype(int)
    m["ungated_unreachable"] = (m["ungated_distance"] < 0).astype(int)
    m["induced_unreachable"] = (m["induced_distance"] < 0).astype(int)
    key = ["arch_id", "source"]
    for col in ("gated_distance", "ungated_distance", "induced_distance"):
        mx = m[m[col] >= 0].groupby(key)[col].max().rename("mx")
        m = m.join(mx, on=key)
        m[col] = np.where(m[col] >= 0, m[col], m["mx"].fillna(0) + 1)
        m = m.drop(columns=["mx"])
    m["log_redundancy"] = np.log1p(np.maximum(m["redundancy"], 0))
    m["log_ungated_redundancy"] = np.log1p(np.maximum(m["ungated_redundancy"], 0))
    m["log_induced_redundancy"] = np.log1p(np.maximum(m["induced_redundancy"], 0))
    m["dist_x_red"] = m["gated_distance"] * m["log_redundancy"]
    m["udist_x_ured"] = m["ungated_distance"] * m["log_ungated_redundancy"]
    m["idist_x_ired"] = m["induced_distance"] * m["log_induced_redundancy"]
    return m.reset_index(drop=True)


def h4_position(ev: pd.DataFrame, design: dict, setup: dict) -> dict:
    """Fitted independently on the evaluation sample with development standardisation and the
    frozen ridge; architecture-clustered bootstrap intervals."""
    lam, sc = setup["lam"], setup["sc"]
    m = fit_node_model(ev, H4_COLS, lam, sc)
    b = m["beta"]
    groups = list(ev.groupby("arch_id").indices.values())
    def refit(ix):
        return fit_node_model(ev.iloc[np.concatenate(ix)], H4_COLS, lam, sc)["beta"]
    boots = boot_over(groups, refit, design["n_boot"], design["boot_seed"])
    coef = {c: dict(estimate=float(b[i + 1]), ci=ci(boots[:, i + 1])) for i, c in enumerate(H4_COLS)}
    crit = dict(a_distance_negative=bool(coef["gated_distance"]["ci"][1] < 0),
                b_redundancy_positive=bool(coef["log_redundancy"]["ci"][0] > 0),
                c_interaction_positive=bool(coef["dist_x_red"]["ci"][0] > 0))
    return dict(coefficients=coef, lam=lam, n_rows=int(len(ev)), n_architectures=int(len(groups)),
                criteria=crit, passed=bool(all(crit.values())))


GATED_FORM = ["gated_distance", "log_redundancy", "dist_x_red"]
INDUCED_FORM = ["induced_distance", "log_induced_redundancy", "idist_x_ired"]
UNGATED_FORM = ["ungated_distance", "log_ungated_redundancy", "udist_x_ured"]


def _eligible(df: pd.DataFrame) -> pd.DataFrame:
    """H6 rows: complete for at least one delivery class available from the source and reachable
    ungated. Any infinite distance is encoded as the eligible set's maximum finite value plus 1."""
    e = df[(df["eligible"] == 1) & (df["ungated_unreachable"] == 0)].copy()
    key = ["arch_id", "source"]
    for col, flag in (("gated_distance", "gated_unreachable"), ("induced_distance", "induced_unreachable")):
        fin = e[e[flag] == 0].groupby(key)[col].max().rename("mx")
        e = e.join(fin, on=key)
        e[col] = np.where(e[flag] == 0, e[col], e["mx"].fillna(0) + 1)
        e = e.drop(columns=["mx"])
    e["dist_x_red"] = e["gated_distance"] * e["log_redundancy"]
    e["idist_x_ired"] = e["induced_distance"] * e["log_induced_redundancy"]
    return e.reset_index(drop=True)


def h6_gate(dev: pd.DataFrame, ev: pd.DataFrame, design: dict, lam: float) -> dict:
    dev_e, ev_e = _eligible(dev), _eligible(ev)
    forms = dict(gated=GATED_FORM, induced=INDUCED_FORM, ungated=UNGATED_FORM)
    models = {k: fit_node_model(dev_e, cols, lam) for k, cols in forms.items()}
    C = {k: within_arch_concordance(ev_e, predict_node_model(m_, ev_e)) for k, m_ in models.items()}
    common = C["gated"].index.intersection(C["induced"].index).intersection(C["ungated"].index)
    point = {k: float(v[common].mean()) for k, v in C.items()}
    dev_groups = list(dev_e.groupby("arch_id").indices.values())
    ev_ids = np.array(common)
    rng = np.random.default_rng(design["boot_seed"])
    bd, bd2 = [], []
    for _ in range(design["n_boot"]):
        dsub = dev_e.iloc[np.concatenate([rng.choice(g, len(g)) for g in dev_groups])]
        ms = {k: fit_node_model(dsub, cols, lam) for k, cols in forms.items()}
        pick = rng.choice(ev_ids, len(ev_ids))
        sub = ev_e[ev_e["arch_id"].isin(set(pick))]
        cb = {k: within_arch_concordance(sub, predict_node_model(m_, sub)) for k, m_ in ms.items()}
        w = pd.Series(pick).value_counts()
        idx = cb["gated"].index.intersection(cb["induced"].index).intersection(cb["ungated"].index)
        if len(idx):
            bd.append(float(np.average(cb["gated"][idx] - cb["induced"][idx], weights=w[idx].values)))
            bd2.append(float(np.average(cb["gated"][idx] - cb["ungated"][idx], weights=w[idx].values)))
    bd, bd2 = np.array(bd), np.array(bd2)
    delta = point["gated"] - point["induced"]
    return dict(c_gated=point["gated"], c_induced=point["induced"], c_ungated=point["ungated"],
                delta_c=delta, ci=ci(bd), delta_c_secondary=point["gated"] - point["ungated"], ci_secondary=ci(bd2),
                n_architectures=int(len(common)), n_rows=int(len(ev_e)), n_excluded=int(len(ev) - len(ev_e)),
                passed=bool(delta >= design["h6_min_delta_c"] and ci(bd)[0] > 0))


# ---- H5 shape ----------------------------------------------------------------------

def h5_shape(df: pd.DataFrame, design: dict) -> dict:
    """H5(a) matched-count blocks as specified: chain below star and below mesh by >= 0.05 (unchanged).
    H5(b) representative blocks: chain below star by >= 0.05, chain below mesh at each degree (CI > 0).
    Blocks are resampled within their (f) or (f, entry share) cell."""
    out, ok = {}, True
    d = df[df["family"] == "matched"]
    piv = d.pivot_table(index=["f_target", "block"], columns="shape", values="Y").reset_index().dropna()
    strata = [g.index.values for _, g in piv.groupby("f_target")]
    def contrast(piv_, strata_, a_, b_):
        v = (piv_[a_] - piv_[b_]).values
        bs = boot_over(strata_, lambda ix: v[np.concatenate(ix)].mean(), design["n_boot"], design["boot_seed"])
        return v, bs
    for other, need in (("matched_star", design["h5_min_delta"]), ("matched_mesh", design["h5_min_delta"])):
        v, bs = contrast(piv, strata, other, "matched_chain")
        lo, hi = ci(bs)
        out[f"a_{other}_minus_chain"] = dict(n_blocks=int(len(v)), mean=float(v.mean()), ci=(lo, hi), min_required=need)
        ok = ok and lo > 0 and v.mean() >= need
    v, bs = contrast(piv, strata, "matched_star", "matched_mesh")
    out["a_star_minus_mesh_reported"] = dict(n_blocks=int(len(v)), mean=float(v.mean()), ci=ci(bs))
    r = df[df["family"] == "representative"]
    if len(r):
        piv = r.pivot_table(index=["f_target", "entry_share", "block"], columns="shape", values="Y").reset_index().dropna()
        strata = [g.index.values for _, g in piv.groupby(["f_target", "entry_share"])]
        cols = [c_ for c_ in piv.columns if str(c_).startswith("rep_") and c_ != "rep_chain"]
        for other in cols:
            need = design["h5_min_delta"] if other == "rep_star" else 0.0
            v, bs = contrast(piv, strata, other, "rep_chain")
            lo, hi = ci(bs)
            out[f"b_{other}_minus_chain"] = dict(n_blocks=int(len(v)), mean=float(v.mean()), ci=(lo, hi), min_required=need)
            ok = ok and lo > 0 and v.mean() >= need
        curve = r.groupby(["shape", "entry_share"])["Y"].mean().unstack("entry_share").round(4)
        out["b_entry_share_curve"] = {str(k): {str(e): float(v_) for e, v_ in row.items()} for k, row in curve.iterrows()}
    return dict(contrasts=out, passed=bool(ok))


# ---- implementation checks (Section 11) --------------------------------------------

def cut_fixtures() -> dict:
    """Paper 1B eq. (10) on single-supply, alternative-supply and shared-upstream fixtures, evaluated
    through the same measure() used for every architecture."""
    c, sp, RC, Cond = _core(), _core().sp, _core().RC, _core().Cond
    def build(rels):
        nodes = [_node(sp, n_, "estate", False, True, True) for n_ in ("u", "a", "b", "t")] + [_source(sp)]
        return make_spec("cutfx", nodes, rels, 1)
    conn = lambda x, y: sp.Relation(x, y, Cond.INTERFACE, RC.CONNECTION)
    single = measure(build([conn("src0", "a"), conn("a", "t")]), ["src0"])[1]["src0"]["t"]["cut_I_src"]
    alt = measure(build([conn("src0", "a"), conn("src0", "b"), conn("a", "t"), conn("b", "t")]), ["src0"])[1]["src0"]["t"]["cut_I_src"]
    shared = measure(build([conn("src0", "u"), conn("u", "a"), conn("u", "b"), conn("a", "t"), conn("b", "t")]), ["src0"])[1]["src0"]["t"]["cut_I_src"]
    ok = single.startswith("a:") and alt == "" and "u:upstream" in shared
    return dict(passed=bool(ok), single_supply=single, alternative_supply=alt or "none", shared_upstream=shared)


def implementation_checks(run_dir: str, arch: pd.DataFrame, nodes: pd.DataFrame, pilot: bool) -> dict:
    res = {}
    ic3 = os.path.join(run_dir, "ic3_deterministic.csv")
    ok3, notes = True, []
    if os.path.exists(ic3):
        d = pd.read_csv(ic3)
        for _, r in d.iterrows():
            reached = set(str(r["reached"]).split(";")) if isinstance(r["reached"], str) else set()
            nn = nodes[nodes["arch_id"] == r["arch_id"]]
            if r["shape"].startswith("chain"):
                want = set(nn[nn["role"].str.startswith("depth")]["node"])
            elif r["variant"] == "hub_removed":
                want = set(nn[nn["role"] == "entry_leaf"]["node"])
            elif r["shape"] == "star_cp":
                # member takeover of a plane is a rate (Layer 3), not structure: in deterministic
                # mode a leaf entry reaches the entry leaf only (frozen v0.8 behaviour, stated)
                want = set(nn[nn["role"] == "entry_leaf"]["node"])
            elif r["shape"] == "star_hubsource":
                want = set(nn[nn["role"].isin(["leaf", "entry_leaf"])]["node"])
            elif r["shape"] == "star_conn":
                want = set(nn[nn["role"].isin(["leaf", "entry_leaf", "hub"])]["node"])
            else:
                want = set(nn[(nn["gated_distance"] >= 0)]["node"])
            good = reached == want
            ok3 = ok3 and good
            if not good:
                notes.append(f"{r['arch_id']}: reached {len(reached)} expected {len(want)}")
    res["IC3_structural_exactness"] = dict(passed=ok3, notes=notes[:10],
                                           note="control-plane star with leaf entry: takeover is a rate, so the deterministic reach is the entry leaf; the hub-source star must reach every member")
    canon = arch[arch["family"] == "canonical"]
    exact = (canon["n_A_assigned"] == (canon["f_target"] * canon["N_target"]).round()).all() if len(canon) else True
    res["IC4_exact_canonical_gate"] = dict(passed=bool(exact), note="A count equals round(fN) before any removal variant")
    mt = arch[arch["family"].isin(["matched", "representative"])]
    ok5 = True
    mn = nodes[nodes["arch_id"].isin(mt["arch_id"])].merge(mt[["arch_id", "block", "family", "n_relations"]], on="arch_id")
    for block, g in mn.groupby("block"):
        sigs = set()
        fam = g["family"].iloc[0]
        for aid, ga in g.groupby("arch_id"):
            sig = tuple(sorted(zip(ga["node"], ga["I_native"], ga["X_native"], ga["A_native"],
                                   ga["x_prob"].round(12), ga["a_prob"].round(12))))
            sigs.add((sig, int(ga["n_relations"].iloc[0]) if fam == "matched" else 0))
        ok5 = ok5 and len(sigs) == 1
    res["IC5_matched_blocks"] = dict(passed=bool(ok5), note="node set, native corners, gate draws and relation count identical within every block")
    exp_counts = pd.Series([f"{t['meta']['sample']}|{t['family']}|{t.get('shape', 'mixed')}|{t.get('cell', '')}|"
                            f"{t.get('variant', '' if t['family'] == 'mixed' else (t['shape'] if t['family'] == 'representative' else 'intact'))}"
                            for t in all_tasks(DESIGN, pilot)]).value_counts()
    got = (arch["sample"] + "|" + arch["family"] + "|" + arch["shape"] + "|" + arch["cell"].astype(str) + "|" + arch["variant"].astype(str)).value_counts()
    missing = {k: int(v) for k, v in exp_counts.items() if got.get(k, 0) != v}
    g = nodes[nodes["gated_distance"] >= 0]
    res["IC6_measures"] = dict(passed=bool((g["gated_distance"] >= g["ungated_distance"]).all() and (g["redundancy"] >= 1).all()
                                           and (nodes["redundancy"] <= nodes["ungated_redundancy"]).all()
                                           and (nodes.loc[nodes["ungated_distance"] < 0, "gated_distance"] < 0).all()),
                               note="also enforced at build; a violation aborts generation")
    samples = arch.groupby("sample")["sample_seed"].unique().to_dict()
    pilot_seeds = set(sum([[DESIGN["pilot_seeds"]["development"], DESIGN["pilot_seeds"]["heldout"]] + DESIGN["pilot_seeds"]["replication"]], []))
    res["IC7_samples"] = dict(passed=bool((pilot or not any(s in pilot_seeds for v in samples.values() for s in v)) and not missing),
                              seeds={k: [int(x) for x in v] for k, v in samples.items()},
                              cells_with_wrong_count=len(missing), examples=dict(list(missing.items())[:5]))
    res["IC2_ceiling"] = dict(passed=True, note="enforced during generation against the source-specific eligible union; any violation aborted the run")
    res["IC6_fixtures"] = ic6_fixtures()
    res["IC6_cut_fixtures"] = cut_fixtures()
    res["IC8_validate_guard"] = dict(passed=True, note="ScenarioSpec.validate is replaced at load by a guard that aborts if called")
    return res


# ---- exploratory, docker, figures -------------------------------------------------------

def exploratory(arch: pd.DataFrame, nodes: pd.DataFrame, pilot: bool) -> pd.DataFrame:
    rows = []
    hs = arch[arch["shape"] == "star_hubsource"]
    for (f, n, N), g in hs.groupby(["f_target", "n", "N_target"]):
        rows.append(dict(section="star_hub_source", key=f"f{f}_N{int(N)}_n{int(n)}_share{n / (N - 1):.2f}", value=float(g["Y"].mean()),
                         note="Y with the source as the plane controller; share = members / (N - 1)"))
    mt = arch[arch["family"] == "matched"]
    for f, g in mt.groupby("f_target"):
        p = g.pivot_table(index="block", columns="shape", values="Y")
        for s in p.columns:
            rows.append(dict(section="matched_Y", key=f"f{f}_{s}", value=float(p[s].mean()), note="mean Y by arrangement"))
    mx = nodes.merge(arch[["arch_id", "family"]], on="arch_id")
    mx = mx[mx["family"] == "mixed"]
    for dist, g in mx.groupby("ungated_distance"):
        a, b = g[(g["boundary_X"] == 1) | (g["boundary_A"] == 1)], g[(g["boundary_X"] == 0) & (g["boundary_A"] == 0)]
        if len(a) >= 20 and len(b) >= 20:
            rows.append(dict(section="boundary_supply", key=f"dist{dist}", value=float(a["p"].mean() - b["p"].mean()),
                             note="p(outside X or A) minus p(inside) at equal ungated distance"))
    pi_c = FIXED["beta_conn"] * FIXED["execution_rate"] * FIXED["authority_rate"]
    for f in DESIGN["f_levels"]:
        q = f * (1 - (1 - pi_c) ** FIXED["max_steps"])
        k = math.ceil(math.log(0.05) / math.log(q)) if 0 < q < 1 else float("inf")
        rows.append(dict(section="diminishing_hops", key=f"f{f}", value=float(k), note="single-path depth at which predicted p falls below 0.05 (connection, horizon 50)"))
    rows.append(dict(section="value_weighted", key="corr_impact_Y", value=float(arch[["impact_reached", "Y"]].corr().iloc[0, 1]), note="correlation of impact reached with Y"))
    top = mx[mx["impact"] >= mx["impact"].quantile(0.9)]
    rows.append(dict(section="value_weighted", key="p_top_decile_impact", value=float(top["p"].mean()), note="mean p of the highest-impact decile of nodes (mixed)"))
    for fam, g in arch.groupby("family"):
        multi = g[g["sources"].astype(str).str.count(";") >= 1]
        rows.append(dict(section="variance_split", key=fam, value=float(multi["var_share_source"].mean()) if len(multi) else float("nan"),
                         note="share of within-architecture variance explained by the source; n/a for single-source families"))
    fc = [c for c in arch.columns if c.startswith("f_") and c[2:] in FACTORS_DEV]
    if fc:
        C = arch[arch["family"] == "mixed"][fc].corr().abs()
        M = C.values - np.eye(len(fc))
        i, j = np.unravel_index(np.nanargmax(M), M.shape)
        rows.append(dict(section="factor_correlations", key="max_abs_offdiag", value=float(M[i, j]), note=f"Latin hypercube caveat; pair {fc[i]} x {fc[j]}"))
    if pilot:
        for m_ in NODE_MEASURES:
            if m_ in mx.columns and mx[m_].std() > 0:
                rows.append(dict(section="pilot_screen_node", key=m_, value=concordance(mx[m_].values, mx["p"].values), note="single-measure concordance with p (mixed)"))
        ma = arch[arch["family"] == "mixed"]
        for m_ in ARCH_MEASURES:
            if m_ in ma.columns and ma[m_].std() > 0:
                rows.append(dict(section="pilot_screen_arch", key=m_, value=concordance(ma[m_].values, ma["Y"].values), note="single-measure concordance with Y (mixed)"))
    return pd.DataFrame(rows)


def write_docker_subsample(run_dir: str, design: dict) -> None:
    """Nine predetermined Docker-scale counterparts at N = 30, f = 0.6, seed 4201 (Section 14)."""
    rows = []
    sdir = os.path.join(run_dir, "docker_h2_specs")
    os.makedirs(sdir, exist_ok=True)
    seed = design["seeds"]["development"]
    N, f = design["docker_n"], design["docker_f"]
    d_star, _ = mesh_prediction(f, N, 0.01)
    grid = mesh_densities(N, f)
    nearest = lambda x: min(grid, key=lambda g: abs(np.log(g) - np.log(x)))
    plan = [("mesh", dict(d=nearest(0.5 * d_star))), ("mesh", dict(d=nearest(d_star))), ("mesh", dict(d=nearest(2 * d_star))),
            ("star_cp", dict(n=20, variant="intact")), ("star_cp", dict(n=20, variant="hub_removed")),
            ("star_conn", dict(n=20, variant="intact")),
            ("chain_conn", dict(depth=12)), ("chain_chan", dict(depth=12, sixth=True)), ("chain_chan", dict(depth=12, sixth=False))]
    for k, (shape, kw) in enumerate(plan):
        aid = f"docker_{k:02d}_{shape}" + (f"_{kw['variant']}" if kw.get("variant") else "") + ("_negcontrol" if kw.get("sixth") is False else "")
        # the matched chain pair shares one seed so the two differ only at the sixth node
        task = dict(arch_id=aid, family="canonical", shape=shape, seed=seed * 1000 + (7 if shape == "chain_chan" else k), N=N, f=f, meta={}, **kw)
        spec, meta, ameta = build_canonical(task)
        reach = _det_reach(spec, "src0")
        with open(os.path.join(sdir, aid + ".yaml"), "w") as fh:
            fh.write(spec_to_yaml(spec))
        rows.append(dict(arch_id=aid, shape=shape, variant=kw.get("variant", ""), neg_control=bool(kw.get("sixth") is False),
                         N=N, f=f, d=kw.get("d", ""), model_reach_set=";".join(reach), n_seeds=design["docker_seeds"],
                         observed_reach_set=""))
    pd.DataFrame(rows).to_csv(os.path.join(run_dir, "docker_h2.csv"), index=False)


# ---- the three generated figures (Section 3.6) ----------------------------------------------

def _tri(cx, cy, r, held, cls="#5F5E5A", open_col="#0F6E56"):
    """A node triangle centred (cx, cy): corners I top, X bottom-left, A bottom-right; held = (I, X, A)."""
    pts = [(cx, cy - r), (cx - 0.87 * r, cy + 0.5 * r), (cx + 0.87 * r, cy + 0.5 * r)]
    L = [f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" stroke="#888780" stroke-width="1"/>']
    for (x, y), h in zip(pts, held):
        L.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{cls if h else "none"}" stroke="{cls if h else open_col}" stroke-width="1.5"/>')
    return "\n".join(L), pts


def write_theory_figures(run_dir: str, figs: str) -> None:
    hdr = '<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="sans-serif" font-size="11"><rect width="{w}" height="{h}" fill="white"/>'
    # 1 the eight-glyph key
    L = [hdr.format(w=640, h=220)]
    kinds = [("native", (1, 1, 1)), ("I", (0, 1, 1)), ("X", (1, 0, 1)), ("A", (1, 1, 0)),
             ("I and X", (0, 0, 1)), ("I and A", (0, 1, 0)), ("X and A", (1, 0, 0)), ("all three", (0, 0, 0))]
    for k, (name, held) in enumerate(kinds):
        cx, cy = 60 + (k % 4) * 150, 50 + (k // 4) * 90
        t, _ = _tri(cx, cy, 24, held, cls="#534AB7" if name == "native" else "#0F6E56")
        L += [t, f'<text x="{cx}" y="{cy + 42}" text-anchor="middle">{name}</text>']
    L.append('<text x="20" y="208">Filled corner: held by the node. Open corner: supplied by another node. I top, X left, A right.</text></svg>')
    open(os.path.join(figs, "fig1_eight_glyph_key.svg"), "w", encoding="utf-8").write("\n".join(L))
    # 2 a network of triangles from the first saved mixed scenario file
    sdir = os.path.join(run_dir, "specs")
    mixed = sorted(f for f in os.listdir(sdir) if "_mixed_" in f)[:1] if os.path.isdir(sdir) else []
    if mixed:
        c = _core()
        raw = open(os.path.join(sdir, mixed[0])).read()
        node_ids = re.findall(r"^- id: (\S+)", raw, flags=re.M)
        govs = dict(zip(node_ids, re.findall(r"^  governance: (\S+)", raw, flags=re.M)))
        rels = re.findall(r"- supplier: (\S+)\n  receiver: (\S+)\n  condition: (\S+)", raw)
        est = [n for n in node_ids if govs.get(n) == "estate"][:24]
        ext = [n for n in node_ids if govs.get(n) != "estate"][:4]
        pos = {}
        for k, n in enumerate(est):
            pos[n] = (70 + (k % 6) * 90, 60 + (k // 6) * 90)
        for k, n in enumerate(ext):
            pos[n] = (640, 60 + k * 90)
        L = [hdr.format(w=720, h=440), '<rect x="20" y="20" width="560" height="400" rx="8" fill="none" stroke="#888780" stroke-dasharray="4 4"/>',
             '<text x="28" y="412">Organisation boundary</text>']
        held = {}
        for n in node_ids:
            m_ = re.search(rf"^- id: {re.escape(n)}\n  governance: \S+\n  interface: (\S+)\n  execution_pathway: (\S+)\n  authority: (\S+)", raw, flags=re.M)
            held[n] = tuple(v == "true" for v in m_.groups()) if m_ else (1, 1, 1)
        corner = {"interface": 0, "execution_pathway": 1, "authority": 2}
        for sup, rec, cond in rels:
            if rec in pos and sup in pos and sup != rec:
                t, pts = _tri(*pos[rec], 16, held[rec])
                x, y = pts[corner.get(cond, 0)]
                col = "#993C1D" if govs.get(sup) != "estate" else "#0F6E56"
                L.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{pos[sup][0]:.1f}" y2="{pos[sup][1]:.1f}" stroke="{col}" stroke-width="1" stroke-dasharray="3 3"/>')
        for n, (x, y) in pos.items():
            t, _ = _tri(x, y, 16, held[n])
            L += [t, f'<text x="{x}" y="{y + 30}" text-anchor="middle">{n}</text>']
        L.append("</svg>")
        open(os.path.join(figs, "fig2_network_of_triangles.svg"), "w", encoding="utf-8").write("\n".join(L))
    # 3 the three shapes
    L = [hdr.format(w=640, h=220)]
    for k, (name, inc) in enumerate((("Mesh", "WannaCry"), ("Star", "CrowdStrike"), ("Chain", "SolarWinds"))):
        cx = 110 + k * 210
        if name == "Mesh":
            P = [(cx - 40, 60), (cx + 40, 60), (cx - 40, 130), (cx + 40, 130)]
            E = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3)]
        elif name == "Star":
            P = [(cx, 95), (cx - 60, 50), (cx + 60, 50), (cx - 60, 140), (cx + 60, 140)]
            E = [(0, 1), (0, 2), (0, 3), (0, 4)]
        else:
            P = [(cx, 45), (cx, 95), (cx, 145)]
            E = [(0, 1), (1, 2)]
        for a_, b_ in E:
            L.append(f'<line x1="{P[a_][0]}" y1="{P[a_][1]}" x2="{P[b_][0]}" y2="{P[b_][1]}" stroke="#0F6E56" stroke-width="1.2" stroke-dasharray="3 3"/>')
        for x, y in P:
            L.append(_tri(x, y, 12, (1, 1, 1))[0])
        L += [f'<text x="{cx}" y="185" text-anchor="middle" font-weight="bold">{name}</text>',
              f'<text x="{cx}" y="202" text-anchor="middle">{inc}</text>']
    L.append("</svg>")
    open(os.path.join(figs, "fig3_three_shapes.svg"), "w", encoding="utf-8").write("\n".join(L))


def cfg_v8_reanalysis(run_dir: str) -> Optional[pd.DataFrame]:
    """Section 15: on the v8 validation files, mean collapse rate by supplied-capability quintile
    against connection density. Written only when RUN_CONFIG['v8_run_dir'] points at the v8 run."""
    v8 = RUN_CONFIG.get("v8_run_dir")
    if not v8 or not os.path.isdir(v8):
        return None
    files = [f for f in os.listdir(v8) if f.startswith("validation_") and f.endswith(".csv")]
    if not files:
        return None
    df = pd.concat([pd.read_csv(os.path.join(v8, f)) for f in files], ignore_index=True)
    if "capability_supplied" not in df or "conn_density_realised" not in df:
        return None
    df["cap_q"] = pd.qcut(df["capability_supplied"].rank(method="first"), 5, labels=False)
    df["dens_q"] = pd.qcut(df["conn_density_realised"].rank(method="first"), 8, labels=False)
    out = df.groupby(["cap_q", "dens_q"]).agg(collapse=("trial_collapse_rate", "mean"), n=("trial_collapse_rate", "size"),
                                              capability=("capability_supplied", "mean"), density=("conn_density_realised", "mean")).reset_index()
    out.to_csv(os.path.join(run_dir, "v8_reanalysis.csv"), index=False)
    return out


def svg_lines(series: Dict[str, Tuple[np.ndarray, np.ndarray]], path: str, xlab: str, ylab: str, logx=False) -> None:
    W, Hh, m = 640, 400, 60
    xs = np.concatenate([np.asarray(v[0], float) for v in series.values()])
    ys = np.concatenate([np.asarray(v[1], float) for v in series.values()])
    if logx:
        xs = np.log10(xs)
    x0, x1 = float(np.nanmin(xs)), float(np.nanmax(xs)); y0, y1 = 0.0, max(float(np.nanmax(ys)), 1e-6)
    sx = lambda x: m + (x - x0) / max(x1 - x0, 1e-9) * (W - 2 * m)
    sy = lambda y: Hh - m - (y - y0) / max(y1 - y0, 1e-9) * (Hh - 2 * m)
    cols = ["#1D4ED8", "#0F6E56", "#993C1D", "#534AB7", "#5F5E5A", "#B45309"]
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="sans-serif" font-size="12">',
         f'<rect width="{W}" height="{Hh}" fill="white"/>',
         f'<line x1="{m}" y1="{Hh-m}" x2="{W-m}" y2="{Hh-m}" stroke="#333"/>',
         f'<line x1="{m}" y1="{m}" x2="{m}" y2="{Hh-m}" stroke="#333"/>',
         f'<text x="{W/2}" y="{Hh-15}" text-anchor="middle">{xlab}</text>',
         f'<text x="15" y="{Hh/2}" text-anchor="middle" transform="rotate(-90 15 {Hh/2})">{ylab}</text>']
    for k, (name, (x, y)) in enumerate(series.items()):
        x = np.log10(np.asarray(x, float)) if logx else np.asarray(x, float)
        pts = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(x, y) if np.isfinite(a) and np.isfinite(b))
        L.append(f'<polyline points="{pts}" fill="none" stroke="{cols[k % len(cols)]}" stroke-width="1.6"/>')
        L.append(f'<text x="{W-m+5}" y="{m+14*k}" fill="{cols[k % len(cols)]}">{name}</text>')
    for t in np.linspace(y0, y1, 5):
        L.append(f'<text x="{m-8}" y="{sy(t)+4}" text-anchor="end">{t:.2f}</text>')
    for t in np.linspace(x0, x1, 5):
        lab = f"{10**t:.3g}" if logx else f"{t:.3g}"
        L.append(f'<text x="{sx(t)}" y="{Hh-m+16}" text-anchor="middle">{lab}</text>')
    L.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# ---- summary --------------------------------------------------------------------------------

def _fmt_ci(c):
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]" if all(np.isfinite(c)) else "[nan, nan]"


def write_summary(run_dir: str, final: dict, by_sample: dict, ic: dict, fz: dict, rec_ok: bool, pilot: bool,
                  n: dict, exp: pd.DataFrame, rec_diffs: Optional[List[str]] = None, seeds: Optional[dict] = None) -> str:
    L = [f"# Paper 4A v{VERSION} summary: {'PILOT (pilot seeds only, no verdicts)' if pilot else 'CONFIRMATORY'}", "",
         f"- Run: {os.path.basename(run_dir)}  |  {fz['note']}  |  record " +
         ("matches" if rec_ok else "DIFFERS in " + ", ".join(rec_diffs or []) + " (analysis re-run under exploratory=True; not citable)"),
         f"- Seeds in this run: {seeds}" if seeds else "",
         f"- Architectures: " + ", ".join(f"{k} {v}" for k, v in n.items()),
         "- Intervals: 95% percentile bootstrap over architectures (cells, bases, blocks or whole architectures with all sources and nodes); never over runs or node rows", "",
         ("## Sample results (pilot seeds; verdicts require 4201, 4202 and 4301, which are untouched)" if pilot else
          "## Verdicts (pass required on 4201, 4202 and 4301 separately)"), ""]
    cols = list(by_sample)
    L += ["| H | Verdict | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    def mark(v):
        return "pass" if v is True else "fail" if v is False else str(v)
    for h in ("H1", "H2", "H3", "H4", "H5", "H6"):
        L.append(f"| {h} | {mark(final.get(h))} | " + " | ".join(mark(by_sample.get(s, {}).get(h, {}).get("passed", "n/a")) for s in cols) + " |")
    L += ["", "## Implementation checks", ""]
    for k, v in ic.items():
        L.append(f"- {k}: {'pass' if v.get('passed') else 'FAIL'}" + (f" ({v['notes'][:3]})" if v.get("notes") else ""))
    L += ["", "## Key estimates by sample", ""]
    for s, v in by_sample.items():
        L.append(f"### {s}")
        h1 = v.get("H1", {})
        if h1:
            L.append(f"- H1 mesh: Spearman(f, midpoint) {h1['spearman_f_midpoint']:+.2f} {_fmt_ci(h1['spearman_ci'])}, CV(midpoint x f) {h1['cv']:.2f}; criteria {h1['criteria']}")
            for f, r in h1["per_f"].items():
                L.append(f"    f={f}: midpoint d {r['midpoint']:.4f} CI [{r['midpoint_ci'][0]:.4f}, {r['midpoint_ci'][1]:.4f}], nominal d* {r['d_star_pred']:.4f} (midpoint/d* {r['midpoint'] / r['d_star_pred']:.2f}; H1 tests scaling with f, not the level), plateau gap {r['plateau_gap']:.3f}, rise {r['rise']}")
        h2 = v.get("H2", {})
        if h2:
            worst = min(h2["cells"].items(), key=lambda kv: kv[1]["ci"][0])
            L.append(f"- H2 star: {sum(c['ci'][0] > 0 for c in h2['cells'].values())}/{len(h2['cells'])} cells with CI lower > 0; weakest {worst[0]} contrast {worst[1]['mean_contrast']:+.3f} {_fmt_ci(worst[1]['ci'])}")
        h3 = v.get("H3", {})
        if h3:
            for k_, c in h3["cells"].items():
                L.append(f"- H3 {k_}: max +step {c['max_positive_step']:.4f} (crit {c['step_critical']:.4f}), max std dev {c['observed_maxdev']:.2f} (crit {c['band_critical']:.2f}), max gap {c['max_abs_gap']:.4f}, pass {c['passed']}" + (" [degenerate: chain broken at the first node in every architecture, all zero]" if c.get('degenerate_all_zero') else ""))
        h4 = v.get("H4", {})
        if h4:
            co = h4["coefficients"]
            L.append(f"- H4 position (Firth, ridge {h4['lam']}; {h4['n_architectures']} architectures, {h4['n_rows']} rows; cluster bootstrap over architectures): distance {co['gated_distance']['estimate']:+.3f} {_fmt_ci(co['gated_distance']['ci'])}, "
                     f"log redundancy {co['log_redundancy']['estimate']:+.3f} {_fmt_ci(co['log_redundancy']['ci'])}, "
                     f"interaction {co['dist_x_red']['estimate']:+.3f} {_fmt_ci(co['dist_x_red']['ci'])}; criteria {h4['criteria']}")
        h5 = v.get("H5", {})
        if h5:
            for k_, c in h5["contrasts"].items():
                if k_ == "b_entry_share_curve":
                    L.append("- H5 entry-share curve (Y by shape): " + "; ".join(f"{sh}: " + ", ".join(f"{e}={y:.3f}" for e, y in row.items()) for sh, row in c.items()))
                else:
                    L.append(f"- H5 {k_}: {c['mean']:+.3f} {_fmt_ci(c['ci'])} (blocks {c['n_blocks']}" + (f", min {c['min_required']}" if 'min_required' in c else "") + ")")
        h6 = v.get("H6", {})
        if h6:
            L.append(f"- H6 gate (primary, induced-subgraph rival): C gated {h6['c_gated']:.3f} vs {h6['c_induced']:.3f}, delta {h6['delta_c']:+.3f} {_fmt_ci(h6['ci'])} (eligible rows {h6['n_rows']}, architectures {h6['n_architectures']})")
            L.append(f"    secondary, whole-graph rival: C {h6['c_ungated']:.3f}, delta {h6['delta_c_secondary']:+.3f} {_fmt_ci(h6['ci_secondary'])}")
        L.append("")
    if len(exp):
        L += ["## Exploratory headlines", ""]
        for sec in ("matched_Y", "star_hub_source", "diminishing_hops", "value_weighted", "variance_split", "factor_correlations", "boundary_supply"):
            g = exp[exp["section"] == sec]
            if len(g):
                L.append(f"- {sec}: " + ", ".join(f"{r.key} {r.value:.3f}" for r in g.itertuples()))
    txt = "\n".join(L) + "\n"
    with open(os.path.join(run_dir, "P4A_summary.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    return txt


# ---- analysis driver -----------------------------------------------------------------------

def analyse(run_dir: str, design: dict, fz: dict, rec_ok: bool, rec_diffs: List[str], workers: int, pilot: bool) -> dict:
    arch = pd.read_csv(os.path.join(run_dir, "architectures.csv"))
    out = pd.read_csv(os.path.join(run_dir, "outcomes.csv"))
    nodes = pd.read_csv(os.path.join(run_dir, "nodes.csv.gz"), low_memory=False,
                        dtype={"cut_I_src": str, "cut_X_src": str, "cut_A_src": str, "role": str})
    for col in ("cut_I_src", "cut_X_src", "cut_A_src", "role"):
        nodes[col] = nodes[col].fillna("")
    arch = arch.merge(out[out["source"] == "pooled"].drop(columns=["source"]), on="arch_id")
    arch["block"] = arch["block"].fillna("")
    for col in ("variant", "cell"):
        arch[col] = arch[col].fillna("")
    n = arch.groupby("family")["arch_id"].count().to_dict()
    print(f"\n{CODE_VERSION}: {len(arch)} architectures, {len(nodes)} node rows")
    ic = implementation_checks(run_dir, arch, nodes, pilot)
    failed = [k for k, v in ic.items() if not v.get("passed")]
    if failed and not pilot:
        raise SystemExit("Implementation checks failed: " + ", ".join(failed))
    if failed:
        print("  PILOT: implementation checks failed: " + ", ".join(failed))
    dev_nodes = _node_frame(nodes, arch[arch["sample"] == "development"])
    lam, curve = cv_ridge(dev_nodes, H4_COLS, design["h4_ridge_grid"], design["h4_cv_folds"], design["model_seed"])
    print(f"  H4 ridge chosen on development by {design['h4_cv_folds']}-fold CV: {lam} (curve {curve})")
    h4_setup = dict(lam=lam, cv_curve={str(k): v for k, v in curve.items()},
                    sc=Standardiser(dev_nodes[H4_COLS].values.astype(float)))
    samples = {"rep4201": ("replication", design["seeds"]["replication"][0]),
               "rep4202": ("replication", design["seeds"]["replication"][1] if len(design["seeds"]["replication"]) > 1 else None),
               "held4301": ("heldout", design["seeds"]["heldout"])}
    if pilot:
        ps = design["pilot_seeds"]
        samples = {f"pilot_rep{ps['replication'][0]}": ("replication", ps["replication"][0]),
                   f"pilot_held{ps['heldout']}": ("heldout", ps["heldout"])}
    by_sample, preds = {}, []
    for label, (role, seed) in samples.items():
        if seed is None:
            continue
        a = arch[(arch["sample"] == role) & (arch["sample_seed"] == seed)]
        if a.empty:
            continue
        print(f"  {label}: H1..H6 on {len(a)} architectures ...", flush=True)
        ev = _node_frame(nodes, a)
        r3, p3 = h3_chain(a, nodes, design, workers)
        preds.append(p3)
        by_sample[label] = dict(H1=h1_mesh(a, design), H2=h2_star(a, design), H3=r3,
                                H4=h4_position(ev, design, h4_setup), H5=h5_shape(a, design),
                                H6=h6_gate(dev_nodes, ev, design, lam))
    mesh = arch[arch["shape"] == "mesh"]
    for _, r in mesh.iterrows():
        d_star, y_pred = mesh_prediction(float(r["f_XA"]), int(r["N"]), float(r["d"]))
        preds.append(pd.DataFrame([dict(arch_id=r["arch_id"], node="", depth=np.nan, quantity="Y",
                                        observed=r["Y"], predicted=y_pred, d_star=d_star)]))
    stars = arch[arch["shape"].isin(["star_cp", "star_conn"]) & (arch["variant"] == "intact")]
    for _, r in stars.iterrows():
        p0 = float(nodes[(nodes["arch_id"] == r["arch_id"]) & (nodes["role"] == "entry_leaf")]["p"].mean()) if len(nodes) else np.nan
        preds.append(pd.DataFrame([dict(arch_id=r["arch_id"], node="", depth=np.nan, quantity="dY_hub",
                                        observed=np.nan, predicted=r["Y"] - p0 / r["N"], d_star=np.nan),
                                   dict(arch_id=r["arch_id"], node="", depth=np.nan, quantity="dY_leaves_k",
                                        observed=np.nan, predicted=scale60(int(r["N_target"]), 5) / r["n"] * r["Y"], d_star=np.nan)]))
    if preds:
        pd.concat(preds).to_csv(os.path.join(run_dir, "predictions.csv"), index=False)
    v8 = cfg_v8_reanalysis(run_dir)
    final = {}
    for h in ("H1", "H2", "H3", "H4", "H5", "H6"):
        if pilot:
            final[h] = "NO VERDICT (pilot seeds only)"
            continue
        reps = [by_sample.get(k, {}).get(h, {}).get("passed") for k in ("rep4201", "rep4202")]
        held = by_sample.get("held4301", {}).get(h, {}).get("passed")
        if all(r is True for r in reps) and held is True:
            final[h] = True
        elif all(r is True for r in reps) and held is False:
            final[h] = "SUPPORTED UNDER REPLICATION, NOT GENERALISED"
        elif len(reps) == 2 and all(r is not None for r in reps) and held is not None:
            final[h] = False
        else:
            final[h] = "INCOMPLETE"
    with open(os.path.join(run_dir, "verdicts.json"), "w") as f:
        json.dump(dict(code_version=CODE_VERSION, spec_version=SPEC_VERSION, code_sha256=code_hash(), pilot=pilot,
                       stage="hypotheses complete; exploratory, docker and figures pending", final=final,
                       h4_ridge=dict(chosen=lam, cv_curve={str(k): v for k, v in curve.items()}),
                       by_sample=by_sample, implementation_checks=ic), f, indent=2, default=str)
    exp = exploratory(arch, nodes, pilot)
    exp.to_csv(os.path.join(run_dir, "exploratory.csv"), index=False)
    if not pilot:
        write_docker_subsample(run_dir, design)
    figs = os.path.join(run_dir, "figures"); os.makedirs(figs, exist_ok=True)
    try:
        write_theory_figures(run_dir, figs)
    except Exception as e:
        print(f"  theory figures failed softly: {e}")
    try:
        for label, v in by_sample.items():
            mesh = arch[(arch["shape"] == "mesh") & (arch["sample_seed"] == samples[label][1])]
            ser = {f"f={f}": (g.groupby("d")["Y"].mean().index.values, g.groupby("d")["Y"].mean().values) for f, g in mesh.groupby("f_target")}
            svg_lines(ser, os.path.join(figs, f"H1_mesh_{label}.svg"), "connection density d (log)", "Y", logx=True)
            ser = {k_: (np.arange(1, len(c["observed"]) + 1), np.array(c["observed"])) for k_, c in v["H3"]["cells"].items()}
            svg_lines(ser, os.path.join(figs, f"H3_chain_{label}.svg"), "depth k", "p_k")
    except Exception as e:
        print(f"  figure generation failed softly: {e}")
    with open(os.path.join(run_dir, "verdicts.json"), "w") as f:
        json.dump(dict(code_version=CODE_VERSION, spec_version=SPEC_VERSION, code_sha256=code_hash(), pilot=pilot,
                       citable=bool(fz["verified"] and rec_ok and not pilot and not failed), final=final,
                       h4_ridge=dict(chosen=lam, cv_curve={str(k): v for k, v in curve.items()}),
                       by_sample=by_sample, implementation_checks=ic,
                       generated=_dt.datetime.now().isoformat(timespec="seconds")), f, indent=2, default=str)
    rec_path = os.path.join(run_dir, "run_record.json")
    with open(rec_path) as f_:
        rec = json.load(f_)
    rec["h4_ridge"] = dict(chosen=lam, cv_curve={str(k): v for k, v in curve.items()})
    with open(rec_path, "w") as f_:
        json.dump(rec, f_, indent=2)
    txt = write_summary(run_dir, final, by_sample, ic, fz, rec_ok, pilot, n, exp, rec_diffs,
                        {k: v for k, v in ic.get("IC7_samples", {}).get("seeds", {}).items()})
    with open(os.path.join(run_dir, "verdicts.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print("\n" + txt)
    print(f"Outputs -> {run_dir}")
    return final


# ############################################################################
# ENTRY POINT
# ############################################################################

MODES = [("PILOT", "pilot", "pilot seeds, reduced budget, broad screen, no verdicts"),
         ("TIME", "time", "time 50 architectures at full runs; writes the budget estimate"),
         ("ALL", "all", "full confirmatory run: generate every sample, then analyse"),
         ("GENERATE", "generate", "generate or resume the confirmatory samples only"),
         ("ANALYSE", "analyse", "analyse an existing run directory")]


def choose_mode() -> str:
    print("Select mode:")
    for k, (label, _, desc) in enumerate(MODES, 1):
        print(f"  {k}  {label:9s} {desc}")
    while True:
        a = input("Enter number or name: ").strip().upper()
        for k, (label, mode, _) in enumerate(MODES, 1):
            if a in (str(k), label):
                return mode
        print("  not recognised")


def choose_run_id(out_root: str) -> str:
    runs = sorted(d for d in os.listdir(out_root)
                  if os.path.exists(os.path.join(out_root, d, "run_record.json"))) if os.path.isdir(out_root) else []
    if not runs:
        raise SystemExit(f"No runs with a run record under {out_root}.")
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
    ap.add_argument("--mode", type=str.lower, choices=[m[1] for m in MODES])
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
    workers = cfg["workers"] or max(1, (os.cpu_count() or 2) - 2)
    print("=" * 70)
    print(f"  {CODE_VERSION}   mode={mode}")
    print("=" * 70)
    fz = freeze_status()
    print(f"  embedded cemt_core {fz['core_version']}: {fz['note']}")
    if not fz["verified"]:
        raise SystemExit("  IC1 failed: hash mismatch, run aborted.")
    if mode == "time":
        est = time_budget(DESIGN, workers)
        os.makedirs(cfg["out_root"], exist_ok=True)
        with open(os.path.join(cfg["out_root"], f"time_estimate_v{VERSION}.json"), "w") as f:
            json.dump(est, f, indent=2)
        return
    if mode == "analyse" and not cfg["run_id"]:
        cfg["run_id"] = choose_run_id(cfg["out_root"])
    pilot = mode == "pilot"
    run_id = cfg["run_id"] or _dt.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_p4a_{mode}_v{VERSION}"
    run_dir = os.path.join(cfg["out_root"], run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"  run directory: {run_dir}")
    rec_path = os.path.join(run_dir, "run_record.json")
    if mode == "analyse" and not os.path.exists(rec_path):
        raise SystemExit("No run record in that directory.")
    if os.path.exists(rec_path):
        pilot = bool(json.load(open(rec_path)).get("pilot", False))   # resume under the recorded design
    rec_ok, diffs = write_or_check_record(run_dir, run_record(DESIGN, fz, pilot))
    if not rec_ok:
        msg = "The current file differs from the run record in: " + ", ".join(sorted(diffs)) + "."
        if not cfg["exploratory"]:
            raise SystemExit(msg + " Stopped. Restore the recorded version or set exploratory=True.")
        print("  WARNING: " + msg + " Exploratory; output is not citable.")
    if mode in ("pilot", "all", "generate"):
        generate(run_dir, DESIGN, workers, pilot, cfg)
    if mode in ("pilot", "all", "analyse"):
        analyse(run_dir, dict(DESIGN, n_boot=DESIGN["pilot_n_boot"]) if pilot else DESIGN, fz, rec_ok, diffs, workers, pilot)
        with open(rec_path) as f:
            rec = json.load(f)
        rec["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
        with open(rec_path, "w") as f:
            json.dump(rec, f, indent=2)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()