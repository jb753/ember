#!/usr/bin/env python3
"""Plot convergence histories and settle-time speedup for duct_cfl_timing.

Reads every .cnv file in a directory with ConvergenceHistory.read_cnv and
writes three figures, each a 2x2 matrix of n_stage family (scree vs RK4) by
sf_resid, with lines shaded by fac_mgrid within a family:

* Zeta: entropy rise against i_step.
* Residual: energy residual (drhoe) against i_step, log scale.
* Speedup: a bar chart of settle time (ConvergenceHistory.find_settling_record)
  normalised to the scree sf_resid=0 fac_mgrid=0.4 baseline.

Every trace marks its settling record (ConvergenceHistory.find_settling_record).
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

from ember.convergence_history import ConvergenceHistory

NAME = re.compile(
    r"s(?P<n_stage>\d+)_sf(?P<sf_resid>[0-9.]+)_fm(?P<fac_mgrid>[0-9.]+)\.cnv$")

# Family base colors (scree / RK4), shaded across fac_mgrid within a family.
BASE_COLOR = {"0": "#2a78d6", "4": "#eb6834"}
SHADES = {"0.0": 1.0, "0.2": 0.7, "0.4": 0.45}

def _shade(hex_color, factor):
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    return (r * factor, g * factor, b * factor)


def load_cases(cnv_dir, tol):
    paths = sorted(glob.glob(os.path.join(cnv_dir, "*.cnv")))
    if not paths:
        sys.exit(f"no .cnv files found in {cnv_dir}")

    cases = []
    for path in paths:
        m = NAME.search(os.path.basename(path))
        if not m:
            continue
        hist = ConvergenceHistory.read_cnv(path)
        idx = hist.find_settling_record(tol=tol)
        cases.append({
            "n_stage": m["n_stage"],
            "sf_resid": m["sf_resid"],
            "fac_mgrid": m["fac_mgrid"],
            "label": (f"s{m['n_stage']} sf{float(m['sf_resid']):g} "
                      f"fm{float(m['fac_mgrid']):g}"),
            "color": _shade(BASE_COLOR[m["n_stage"]], SHADES[m["fac_mgrid"]]),
            "hist": hist,
            "settle_idx": idx,
            "settle_ms": float(hist.time[idx] - hist.time[0]),
        })
    cases.sort(key=lambda c: c["label"])
    return cases


def _mark_settle(ax, hist, idx, y, color):
    ax.plot(hist.i_step[idx], y[idx], "o", color=color, markeredgecolor="white",
            markersize=6, zorder=3)


ROW_TITLE = {"0": "scree (n_stage=0)", "4": "RK4 (n_stage=4)"}


def _plot_matrix(cases, y_of, ylabel, title, semilogy, out):
    """2x2 grid of i_step traces, rows=n_stage family, cols=sf_resid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True,
                              sharex="row")
    ax_of = {
        ("0", "0.0"): axes[0, 0],
        ("0", "1.0"): axes[0, 1],
        ("4", "0.0"): axes[1, 0],
        ("4", "1.0"): axes[1, 1],
    }
    for (n_stage, sf_resid), ax in ax_of.items():
        ax.set_title(f"{ROW_TITLE[n_stage]}, sf_resid={float(sf_resid):g}",
                     fontsize=10)

    for c in cases:
        hist = c["hist"]
        idx = c["settle_idx"]
        y = y_of(hist)
        ax = ax_of[(c["n_stage"], c["sf_resid"])]
        plot_fn = ax.semilogy if semilogy else ax.plot
        plot_fn(hist.i_step, y, color=c["color"], label=c["label"])
        _mark_settle(ax, hist, idx, y, c["color"])

    for (n_stage, sf_resid), ax in ax_of.items():
        if n_stage == "4":
            ax.set_xlabel("i_step")
        if sf_resid == "0.0":
            ax.set_ylabel(ylabel)

    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    fig.legend(handles, labels, loc="outside right upper", fontsize=8)
    fig.suptitle(title)

    fig.savefig(out, dpi=110)
    print(f"Wrote {out}")


def plot_zeta(cases, out):
    _plot_matrix(cases, lambda hist: hist.zeta, r"$\zeta$ (entropy rise)",
                 "Entropy rise vs step", semilogy=False, out=out)


def plot_residual(cases, out):
    _plot_matrix(cases, lambda hist: hist.residual[:, 4],
                 "energy residual (drhoe)", "Residual decay vs step",
                 semilogy=True, out=out)


def plot_mdot_err(cases, out):
    _plot_matrix(cases, lambda hist: hist.err_mdot,
                 r"$\mathrm{err\_mdot}$", "Mass flow error vs step",
                 semilogy=False, out=out)


def plot_speedup(cases, out, tol):
    base = next((c for c in cases if c["n_stage"] == "0"
                 and float(c["sf_resid"]) == 0.0
                 and float(c["fac_mgrid"]) == 0.4), None)
    if base is None:
        sys.exit("baseline (n_stage=0 sf_resid=0.0 fac_mgrid=0.4) not found")
    base_ms = base["settle_ms"]

    rows = sorted(cases, key=lambda c: base_ms / c["settle_ms"], reverse=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    y = range(len(rows))
    speedups = [base_ms / c["settle_ms"] for c in rows]
    ax.barh(y, speedups, color=[c["color"] for c in rows], height=0.65)
    ax.set_yticks(list(y))
    ax.set_yticklabels([c["label"] for c in rows])
    ax.invert_yaxis()
    ax.axvline(1.0, color="0.4", linewidth=1, linestyle="--")
    ax.set_xlabel("Speedup vs scree sf_resid=0 fac_mgrid=0.4 (settle time, "
                   f"tol={tol:g})")
    ax.set_title("duct_cfl_timing: settle-time speedup by scheme")
    for yi, s in zip(y, speedups):
        ax.text(s + 0.02, yi, f"{s:.2f}x", va="center", fontsize=9)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BASE_COLOR["0"], label="scree (n_stage=0)"),
        plt.Rectangle((0, 0), 1, 1, color=BASE_COLOR["4"], label="RK4 (n_stage=4)"),
    ]
    ax.legend(handles=handles, loc="lower right")

    fig.savefig(out, dpi=110)
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cnv_dir", nargs="?", default="duct_cfl_timing_cnv")
    ap.add_argument("--out-zeta", default="duct_cfl_timing_zeta.pdf")
    ap.add_argument("--out-residual", default="duct_cfl_timing_residual.pdf")
    ap.add_argument("--out-mdot-err", default="duct_cfl_timing_mdot_err.pdf")
    ap.add_argument("--out-speedup", default="duct_cfl_timing_speedup.pdf")
    ap.add_argument("--tol", type=float, default=0.05,
                     help="find_settling_record band, as a fraction of "
                          "zeta's swing (default 0.05)")
    args = ap.parse_args()

    cases = load_cases(args.cnv_dir, args.tol)
    plot_zeta(cases, args.out_zeta)
    plot_residual(cases, args.out_residual)
    plot_mdot_err(cases, args.out_mdot_err)
    plot_speedup(cases, args.out_speedup, args.tol)


if __name__ == "__main__":
    main()
