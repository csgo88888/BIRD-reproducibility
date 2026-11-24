# -*- coding: utf-8 -*-
"""
Plot Figure 6 (paper Section 4.2.2, correlation-centre scan):
accuracy / time / memory vs the correlation centre at fixed gap Delta = 0.20.

Input : results/fig6_center_scan.csv (paper-reported values, shipped), or a
        freshly generated results_generated/fig6_center_scan.csv from
        experiments/fig6_center_scan.py (override with --csv).
Output: figures/fig6_center_{acc,time,mem}.png (single panels) and
        figures/fig6_merged.png (1x3, the layout used in the paper).
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODS = ['BIRD', 'ERM', 'IRM', 'VREx', 'IB-IRM']
STYLE = {
    'BIRD':   dict(label='BIRD (ours)', marker='o', ls='-',  lw=2.2, color='black', mfc='black'),
    'ERM':    dict(label='ERM',         marker='v', ls=':',  lw=1.6, color='0.35',  mfc='0.35'),
    'IRM':    dict(label='IRM',         marker='s', ls='--', lw=1.6, color='0.55',  mfc='white'),
    'VREx':   dict(label='VREx',        marker='^', ls='-.', lw=1.6, color='0.55',  mfc='0.55'),
    'IB-IRM': dict(label='IB-IRM',      marker='D', ls='-.', lw=1.6, color='black', mfc='white'),
}


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    centers = [float(r['center']) for r in rows]
    order = np.argsort(centers)
    data = {'centers': [f'{centers[i]:g}' for i in order]}
    for m in METHODS:
        key = m.replace('-', '_')
        data[m] = {
            'acc':  np.array([float(r[f'{key}_acc']) for r in rows])[order],
            'time': np.array([float(r[f'{key}_time_s']) for r in rows])[order],
            'mem':  np.array([float(r[f'{key}_mem_MBxruns']) for r in rows])[order],
        }
    return data


def draw_panel(ax, data, metric, ylabel, legend=False):
    x = list(range(len(data['centers'])))          # equally spaced categories
    for m in METHODS:
        st = STYLE[m]
        ax.plot(x, data[m][metric], ms=5, mfc=st['mfc'],
                marker=st['marker'], ls=st['ls'], lw=st['lw'], color=st['color'],
                label=st['label'])
    ax.set_xticks(x)
    ax.set_xticklabels(data['centers'], fontsize=8)
    ax.set_xlim(-0.3, len(x) - 0.7)
    ax.minorticks_off()
    ax.tick_params(labelsize=8)
    ax.set_xlabel('correlation centre', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if metric == 'acc':
        ax.set_ylim(0.2, 0.7)
    else:
        ax.set_yscale('log')
    if legend:
        ax.legend(fontsize=6.5, loc='lower right', framealpha=0.9, handlelength=2.2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--csv', default=None, help='input CSV (default: results/fig6_center_scan.csv)')
    ap.add_argument('--out', default=os.path.join(REPO_ROOT, 'figures'), help='output directory')
    a = ap.parse_args()
    path = a.csv or os.path.join(REPO_ROOT, 'results', 'fig6_center_scan.csv')
    if not os.path.exists(path):
        alt = os.path.join(REPO_ROOT, 'results_generated', 'fig6_center_scan.csv')
        path = alt if os.path.exists(alt) else path
    if not os.path.exists(path):
        raise SystemExit(f'CSV not found: {path}\nRun experiments/fig6_center_scan.py first, '
                         f'or pass --csv.')
    os.makedirs(a.out, exist_ok=True)
    data = load(path)

    panels = [('acc', 'worst-env acc.'), ('time', 'time (s)'), ('mem', 'memory (MB x runs)')]
    for metric, ylabel in panels:
        fig, ax = plt.subplots(figsize=(2.4, 2.6), dpi=300)
        draw_panel(ax, data, metric, ylabel, legend=(metric == 'acc'))
        fig.tight_layout()
        out = os.path.join(a.out, f'fig6_center_{metric}.png')
        fig.savefig(out, bbox_inches='tight')
        plt.close(fig)
        print('saved', out)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), dpi=300)
    for ax, (metric, ylabel) in zip(axes, panels):
        draw_panel(ax, data, metric, ylabel)
    handles = [plt.Line2D([], [], ms=5, **{k: v for k, v in STYLE[m].items() if k != 'label'})
               for m in METHODS]
    fig.legend(handles, [STYLE[m]['label'] for m in METHODS], ncol=5, fontsize=8,
               loc='upper center', frameon=True, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    out = os.path.join(a.out, 'fig6_merged.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print('saved', out)


if __name__ == '__main__':
    main()
