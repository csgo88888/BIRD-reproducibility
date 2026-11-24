# -*- coding: utf-8 -*-
"""
Plot Figure 8 (paper Section 4.2.5, scalability study): accuracy / time / memory
vs input dimension d = 2*side^2.

Input : results/fig8_scalability.csv (paper-reported raw per-run rows, shipped),
        or a freshly generated results_generated/fig8_scalability_raw.csv from
        experiments/fig8_scalability.py (override with --csv).
Output: figures/fig8_scalability_{acc,time,mem}.png (three single panels, the
        size used per panel in the paper).

Projection disclosure (must be kept in the caption):
- side = 1400 is NOT measured (float32 data ~ 235 GB, ~33 h per CPU run).
- Time/memory panels: the value at 1400 is a linear-in-d projection through
  the two measured points (training FLOPs and dataset size are both exactly
  proportional to d), drawn as a dotted line + hollow marker with a grey
  "projected" region.
- Accuracy panel: the 1400 point is a flat extension of the 140 measurement
  (dotted + hollow) -- justified by the function-space equivalence verified in
  experiments/mechanism_gradient_binding.py and mechanism_layered_lr.py (under
  nearest-neighbour replication + layered lr, the deployment trajectory at any
  resolution coincides with the 14x14 trajectory; measured 0.626~0.664 at
  k = 4/16/100 vs 0.633~0.654 at 14x14), NOT a two-point trend fit. ERM is
  likewise flat-extended (conservative; its worst-env accuracy is pinned by
  the spurious feature).

Curve conventions (consistent with Figures 5/7): diagnosis = diagnosis time /
peak memory; BIRD (diag+deploy) = diagnosis + two-seed deployment sum;
grid search (12 runs) = 12 x one official ERM run (the equivalent of the
Figure 5 grid baseline; 12 x 17.9 s ~ 215 s at 14x14, consistent with the
measured 209-265 s of Figure 5).

Note (2026-11-23 decision): each Figure 8 panel keeps only the BIRD curve
(acc = BIRD deployment accuracy; time/mem = full BIRD pipeline diag+deploy);
the ERM / grid-search comparison numbers stay in the main text.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIDES_M = [14, 140]              # measured
SIDE_P = 1400                    # projected point
SIDES_ALL = SIDES_M + [SIDE_P]
D = {s: 2 * s * s for s in SIDES_ALL}


def load(path):
    agg = {}
    for l in open(path):
        if not l.strip():
            continue
        side, method, seed, role, acc, dt, mem, lam = l.strip().split(',')
        a = agg.setdefault((method, int(side)), {'acc': [], 'time': 0.0, 'mem': 0.0,
                                                 'runs': 0, 'diag_time': 0.0, 'diag_mem': 0.0})
        if role == 'deploy':
            a['acc'].append(float(acc))
            a['time'] += float(dt.rstrip('s'))
            a['mem'] += float(mem.rstrip('MB'))
            a['runs'] += 1
        else:
            a['diag_time'] += float(dt.rstrip('s'))
            a['diag_mem'] += float(mem.rstrip('MB'))
    return agg


def project(y14, y140):
    """Cost linear in d: value at d(1400) of the line through the two measured points."""
    d1, d2, d3 = D[14], D[140], D[1400]
    slope = (y140 - y14) / (d2 - d1)
    return y14 + slope * (d3 - d1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--csv', default=None, help='input CSV (default: results/fig8_scalability.csv)')
    ap.add_argument('--out', default=os.path.join(REPO_ROOT, 'figures'), help='output directory')
    a = ap.parse_args()
    path = a.csv or os.path.join(REPO_ROOT, 'results', 'fig8_scalability.csv')
    if not os.path.exists(path):
        alt = os.path.join(REPO_ROOT, 'results_generated', 'fig8_scalability_raw.csv')
        path = alt if os.path.exists(alt) else path
    if not os.path.exists(path):
        raise SystemExit(f'CSV not found: {path}\nRun experiments/fig8_scalability.py first, '
                         f'or pass --csv.')
    os.makedirs(a.out, exist_ok=True)

    agg = load(path)
    acc = {m: [np.mean(agg[(m, s)]['acc']) for s in SIDES_M] for m in ['bird', 'erm']}
    diag_t = [agg[('bird', s)]['diag_time'] for s in SIDES_M]
    diag_m = [agg[('bird', s)]['diag_mem'] for s in SIDES_M]
    pipe_t = [agg[('bird', s)]['diag_time'] + agg[('bird', s)]['time'] for s in SIDES_M]
    pipe_m = [agg[('bird', s)]['diag_mem'] + agg[('bird', s)]['mem'] for s in SIDES_M]
    grid_t = [12 * agg[('erm', s)]['time'] / agg[('erm', s)]['runs'] for s in SIDES_M]
    grid_m = [12 * agg[('erm', s)]['mem'] / agg[('erm', s)]['runs'] for s in SIDES_M]

    curves = {
        'time': {'diag': diag_t, 'pipe': pipe_t, 'grid': grid_t},
        'mem':  {'diag': diag_m, 'pipe': pipe_m, 'grid': grid_m},
    }
    proj = {panel: {k: project(v[0], v[1]) for k, v in cs.items()} for panel, cs in curves.items()}

    print('=== projection to 1400 (for the main text) ===')
    for panel in ['time', 'mem']:
        for k in ['diag', 'pipe', 'grid']:
            v = curves[panel][k]
            print(f"{panel}/{k}: measured {v[0]:.3g} -> {v[1]:.3g}, projected@1400 = {proj[panel][k]:.3g}")
    print('measured ratio time(140)/time(14): pipe =', pipe_t[1] / max(pipe_t[0], 1e-9),
          '(theory 100; deviation is fixed overhead)')

    def decorate(ax, ylabel):
        ax.set_xscale('log')
        ax.set_xlim(10, 2600)
        ax.set_xticks(SIDES_ALL)
        # x axis is the input dimension d = 2*side^2 (tick positions unchanged)
        ax.set_xticklabels(['$3.92\\times10^{2}$', '$3.92\\times10^{4}$', '$3.92\\times10^{6}$'], fontsize=7)
        ax.minorticks_off()
        ax.axvspan(140, 1400, color='black', alpha=0.06)
        ax.text(320, 0.03, 'projected', fontsize=5.5, color='0.35', ha='center', va='bottom',
                transform=blended_transform_factory(ax.transData, ax.transAxes))
        ax.tick_params(labelsize=7)
        ax.set_xlabel('input dimension d', fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=7.5)

    def draw_acc():
        fig, ax = plt.subplots(figsize=(2.0, 2.6), dpi=300)
        v = acc['bird']
        ax.plot(SIDES_M, v, color='black', ms=4, mfc='black', marker='o', ls='-', lw=2.0)
        ax.plot([SIDES_M[-1], SIDE_P], [v[-1], v[-1]], color='black',
                ms=4, mfc='white', marker='o', ls=':', lw=1.2, markevery=[1])
        decorate(ax, 'worst-env acc.')
        ax.set_ylim(0.5, 0.70)
        fig.tight_layout()
        out = os.path.join(a.out, 'fig8_scalability_acc.png')
        fig.savefig(out, bbox_inches='tight')
        plt.close(fig)
        print('saved', out)

    def draw_cost(panel, ylabel):
        fig, ax = plt.subplots(figsize=(2.0, 2.6), dpi=300)
        v = curves[panel]['pipe']
        ax.plot(SIDES_M, v, color='black', ms=4, mfc='black', marker='o', ls='-', lw=2.0)
        ax.plot([SIDES_M[-1], SIDE_P], [v[-1], proj[panel]['pipe']], color='black',
                ms=4, mfc='white', marker='o', ls=':', lw=1.2, markevery=[1])
        ax.set_yscale('log')
        decorate(ax, ylabel)
        fig.tight_layout()
        out = os.path.join(a.out, f'fig8_scalability_{panel}.png')
        fig.savefig(out, bbox_inches='tight')
        plt.close(fig)
        print('saved', out)

    draw_acc()
    draw_cost('time', 'time (s)')
    draw_cost('mem', 'memory (MB x runs)')


if __name__ == '__main__':
    main()
