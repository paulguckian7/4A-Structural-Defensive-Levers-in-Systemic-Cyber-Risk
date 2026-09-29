# make_h5_figure.py  (2026-09-27)
# Draws the Paper 4A H5 entry-share figure from a confirmatory run's verdicts.json.
# PhDPaper4A.py writes the curve only as numbers (by_sample -> H5 -> contrasts ->
# b_entry_share_curve); this turns them into a three-panel vector PDF sized for
# IEEEtran full page width, matching H1_mesh_panels.pdf and H3_chain_panels.pdf.
# It also prints the numbers the paper's H5 text relies on, so they can be checked.
#
# Usage (from the folder holding this script):
#   python make_h5_figure.py "<run folder>" "<Images folder>"
# e.g.
#   python make_h5_figure.py "..\cemt_runs\20260920_215757_p4a_all_v12" "..\Images"
# The run folder is the one containing verdicts.json. Needs matplotlib.

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run_dir = sys.argv[1] if len(sys.argv) > 1 else "."
img_dir = sys.argv[2] if len(sys.argv) > 2 else "Images"
path = run_dir if run_dir.endswith(".json") else os.path.join(run_dir, "verdicts.json")

plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})

SHAPES = [("rep_mesh12", "Mesh, degree 12", "#534AB7", "-"),
          ("rep_mesh6", "Mesh, degree 6", "#1D4ED8", "-"),
          ("rep_star", "Star", "#993C1D", "--"),
          ("rep_chain", "Chain", "#0F6E56", ":")]
PANELS = [("rep4201", "Replication 4201"), ("rep4202", "Replication 4202"),
          ("held4301", "Held-out 4301")]

with open(path, encoding="utf-8") as f:
    by_sample = json.load(f)["by_sample"]

fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3), sharey=True)
for ax, (label, title) in zip(axes, PANELS):
    curve = by_sample[label]["H5"]["contrasts"]["b_entry_share_curve"]
    print(f"\n{title}")
    for key, name, col, ls in SHAPES:
        if key not in curve:
            continue
        pts = sorted((float(e), float(y)) for e, y in curve[key].items())
        xs, ys = [100 * e for e, _ in pts], [y for _, y in pts]
        ax.plot(xs, ys, color=col, ls=ls, lw=1.2, marker="o", ms=2.5, label=name)
        steps = ", ".join(f"{xs[i]:g}->{xs[i+1]:g}%: {ys[i+1]-ys[i]:+.3f}" for i in range(len(xs) - 1))
        print(f"  {name:16s} Y = " + ", ".join(f"{y:.3f}" for y in ys) + f"   steps {steps}")
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Entry share (% of nodes)")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25, lw=0.4)
axes[0].set_ylabel("$Y$")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0.09, 1, 1))
os.makedirs(img_dir, exist_ok=True)
out = os.path.join(img_dir, "H5_entry_share.pdf")
fig.savefig(out)
print(f"\nwritten {out}")
