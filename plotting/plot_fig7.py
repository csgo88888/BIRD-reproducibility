# -*- coding: utf-8 -*-
"""
Plot Figure 7 (paper Section 4.2.4, ablation): accuracy / time / memory
across Delta in {0.10, 0.20, 0.30}.

Input : results/fig7_ablation.csv (paper-reported raw per-run rows, shipped),
        or a freshly generated results_generated/fig7_ablation_raw.csv from
        experiments/fig7_ablation.py (override with --csv).
Output: figures/fig7_ablation_{acc,time,mem}.png (three single panels, the
        size used per panel in the paper).
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAPS = [0.10, 0.20, 0.30]
X = [0, 1, 2]  # equally spaced categories: 0.10/0.20/0.30
VARIANTS = ['full', 'no_align', 'no_center', 'no_scalelaw', 'no_safety']
STYLE = {
    'full':        dict(label='BIRD (full)',         marker='o', ls='-',  lw=2.0, mfc='black'),
    'no_align':    dict(label='w/o label alignment', marker='v', ls=':',  lw=1.4, mfc='white'),
    'no_center':   dict(label='w/o centering',       marker='s', ls='--', lw=1.4, mfc='white'),
    'no_scalelaw': dict(label='w/o scaling law',     marker='^', ls='-.', lw=1.4, mfc='white'),
    'no_safety':   dict(label='w/o safety factor',   marker='D', ls='-',  lw=1.2, mfc='white'),
}


def load_csv(path):
    agg = {}
    for l in open(path):
        if not l.strip():
            continue
        v, gap, seed, role, acc, dt, mem, lam = l.strip().split(',')
        a = agg.setdefault((v, float(gap)), {'acc': [], 'time': 0.0, 'mem': 0.0})
        if role == 'deploy':
            a['acc'].append(float(acc))
        a['time'] += float(dt.rstrip('s'))
        a['mem'] += float(mem.rstrip('MB'))
    return agg


def draw(metric, ylabel, series, fname):
    fig, ax = plt.subplots(figsize=(2.0, 2.6), dpi=300)
    for v in VARIANTS:
        st = STYLE[v]
        ax.plot(X, series[v], color='black', ms=3.5, mfc=st['mfc'],
                marker=st['marker'], ls=st['ls'], lw=st['lw'], label=st['label'])
    ax.set_xticks(X)
    ax.set_xticklabels(['0.10', '0.20', '0.30'], fontsize=7)
    ax.set_xlim(-0.3, 2.3)
    ax.minorticks_off()
    if metric == 'time':
        ax.set_ylim(0.0, 25.0)
    if metric == 'mem':
        ax.set_ylim(0.0, 15.0)
    ax.tick_params(labelsize=7)
    ax.set_xlabel(r'environment gap $\Delta$', fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=7.5)
    if metric == 'acc':
        ax.set_ylim(0.15, 0.75)
        ax.legend(fontsize=5.2, loc='lower right', framealpha=0.9, handlelength=1.8)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print('saved', fname)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--csv', default=None, help='input CSV (default: results/fig7_ablation.csv)')
    ap.add_argument('--out', default=os.path.join(REPO_ROOT, 'figures'), help='output directory')
    a = ap.parse_args()
    path = a.csv or os.path.join(REPO_ROOT, 'results', 'fig7_ablation.csv')
    if not os.path.exists(path):
        alt = os.path.join(REPO_ROOT, 'results_generated', 'fig7_ablation_raw.csv')
        path = alt if os.path.exists(alt) else path
    if not os.path.exists(path):
        raise SystemExit(f'CSV not found: {path}\nRun experiments/fig7_ablation.py first, '
                         f'or pass --csv.')
    os.makedirs(a.out, exist_ok=True)

    agg = load_csv(path)
    get = lambda key: {v: [np.mean(agg[(v, g)]['acc']) if key == 'acc'
                           else agg[(v, g)][key] for g in GAPS] for v in VARIANTS}
    draw('acc', 'worst-env acc.', get('acc'), os.path.join(a.out, 'fig7_ablation_acc.png'))
    draw('time', 'time (s)', get('time'), os.path.join(a.out, 'fig7_ablation_time.png'))
    draw('mem', 'memory (MB x runs)', get('mem'), os.path.join(a.out, 'fig7_ablation_mem.png'))


if __name__ == '__main__':
    main()
