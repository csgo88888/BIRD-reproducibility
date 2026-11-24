# -*- coding: utf-8 -*-
"""
Residual scan (paper Section 4.2.1, Figure 5): accuracy / time / memory
========================================================================
ColoredMNIST 14x14, N=5000 per environment, gaps Delta in {0.05, 0.10, 0.20,
0.30} (spurious correlations p1/p2 below), which sweeps the estimated residual
r_hat^2 from ~2.4 to ~119.5.

Two families of rows:
  BIRD (one-off)   r^2 diagnosis + anchored calibration + a single deployment
                   at lambda = 2*C0/r^2, with C0 = lambda_crit(anchor) * r^2(anchor)
                   and anchor Delta = 0.20 (lambda_crit = 1000, measured on the
                   grid below); diagnosis fixed at seed 0, deployment seeds {0, 1}
  grid baselines   official DomainBed classes ERM / IRM / VREx / IB_IRM, with
                   penalty lambda swept over {0.1, 1, 10, 100, 1000, 10000}
                   (ERM has no penalty: lambda = 0 only), seeds {0, 1};
                   the reported baseline accuracy is the best grid point

Every row records worst-environment test accuracy, wall time and peak memory;
each run is executed in a separate subprocess so the memory protocol (per-run
peak) is measured cleanly, as reported in Section 4.1.4. Checkpoint resume:
completed (stage, gap, lambda, seed) rows are skipped.

The DomainBed baselines require DomainBed (official classes, same source as
all figures):
  git clone --depth 1 https://github.com/facebookresearch/DomainBed.git
Place it in the repository root or point the environment variable
DOMAINBED_PATH at it. BIRD-only runs do not need DomainBed.

Usage:
  python experiments/fig5_residual_scan.py                 # full run (a few CPU-hours)
  python experiments/fig5_residual_scan.py --smoke        # Delta=0.20, lambda in {100, 1000}, seed 0
  python experiments/fig5_residual_scan.py --data-dir D --output-dir O
  python experiments/fig5_residual_scan.py --no-baselines # BIRD rows only (no DomainBed needed)

Outputs (in --output-dir):
  fig5_runs.csv           raw per-run rows: stage,gap,lambda,seed,acc,time,mem
  fig5_residual_scan.csv  aggregated table in the exact column layout reported
                          in the paper (results/fig5_residual_scan.csv)
"""
import argparse
import gc
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GAPS = {0.05: (0.825, 0.875), 0.10: (0.8, 0.9), 0.20: (0.75, 0.95), 0.30: (0.7, 1.0)}
N_PER_ENV, STEPS, LR = 5000, 400, 1e-3
ANNEAL_START, ANNEAL_LEN = 150, 100
ANCHOR_GAP = 0.20
LAMBDA_CRIT_MEASURED = {0.10: 10000, 0.20: 1000, 0.30: 100}   # grid-measured critical penalties
ALGOS = ['IRM', 'VREx', 'IB_IRM']
LAM_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
SEEDS = [0, 1]
LAM_KEY = {'IRM': 'irm_lambda', 'VREx': 'vrex_lambda', 'IB_IRM': 'ib_lambda'}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get('BIRD_DATA_DIR') or os.path.join(REPO_ROOT, 'data')
OUT_DIR = os.environ.get('BIRD_OUT_DIR') or os.path.join(REPO_ROOT, 'results_generated')
CSV = os.path.join(OUT_DIR, 'fig5_runs.csv')
TABLE = os.path.join(OUT_DIR, 'fig5_residual_scan.csv')

TR_X = TR_Y = TE_X = TE_Y = None  # lazily loaded (only in worker subprocesses)
_DB = {}


def _load_mnist():
    global TR_X, TR_Y, TE_X, TE_Y
    if TR_X is None:
        from torchvision.datasets import MNIST
        tr = MNIST(DATA_DIR, train=True, download=True)
        te = MNIST(DATA_DIR, train=False, download=True)
        TR_X, TR_Y = tr.data.numpy(), tr.targets.numpy()
        TE_X, TE_Y = te.data.numpy(), te.targets.numpy()


def _domainbed():
    """Lazy import of DomainBed (only needed for the official baselines)."""
    if not _DB:
        for _p in [os.environ.get('DOMAINBED_PATH'), os.path.join(REPO_ROOT, 'DomainBed')]:
            if _p and os.path.isdir(_p) and _p not in sys.path:
                sys.path.insert(0, _p)
        try:
            from domainbed import hparams_registry, algorithms
        except ImportError:
            raise SystemExit(
                'DomainBed not found. Clone it into the repository root or set '
                'DOMAINBED_PATH:\n'
                '  git clone --depth 1 https://github.com/facebookresearch/DomainBed.git\n'
                '(also requires `pip install timm`; or run with --no-baselines)')
        _DB['hparams_registry'] = hparams_registry
        _DB['algorithms'] = algorithms
    return _DB['hparams_registry'], _DB['algorithms']


def peak_mb():
    """Peak RSS of the current process, in MB (cross-platform)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:  # Windows
        import psutil
        info = psutil.Process().memory_info()
        peak = getattr(info, 'peak_wset', None)
        return (peak if peak is not None else info.rss) / 1048576.0


# ---------- data construction (identical to fig7_ablation.py) ----------
def make_env(images, digits, p_color, rng):
    y = (digits < 5).astype(np.float32)
    y = np.where(rng.random(len(y)) < 0.25, 1 - y, y)
    color = np.where(rng.random(len(y)) < p_color, y, 1 - y)
    img = F.avg_pool2d(torch.from_numpy(images).float().unsqueeze(1) / 255., 2).squeeze(1)
    X = torch.zeros(len(y), 2, 14, 14)
    c = torch.from_numpy(color)
    X[:, 0] = img * c[:, None, None]
    X[:, 1] = img * (1 - c)[:, None, None]
    return X.reshape(len(y), -1), torch.from_numpy(y)


def build(gap, seed):
    rng = np.random.default_rng(123)
    perm = rng.permutation(60000)
    te_perm = rng.permutation(10000)
    erng = np.random.default_rng(1000 * seed + int(gap * 100))
    p1, p2 = GAPS[gap]
    envs = [make_env(TR_X[perm[:N_PER_ENV]], TR_Y[perm[:N_PER_ENV]], p1, erng),
            make_env(TR_X[perm[N_PER_ENV:2*N_PER_ENV]], TR_Y[perm[N_PER_ENV:2*N_PER_ENV]], p2, erng)]
    tests = [make_env(TE_X[te_perm[:2500]], TE_Y[te_perm[:2500]], 0.1, erng),
             make_env(TE_X[te_perm[2500:5000]], TE_Y[te_perm[2500:5000]], 0.9, erng)]
    return envs, tests


def diagnostic_r2(envs):
    m_e = np.array([((X[:, :196] - X[:, 196:]).sum(1) * (2 * y - 1)).mean().item()
                    for X, y in envs])
    return float(((m_e - m_e.mean())**2).sum())


# ---------- BIRD deployment (annealed IRMv1 penalty, closed-form gradient) ----------
def train_ours(envs, lam, seed):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(392, 128), nn.ReLU(), nn.Linear(128, 1))
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    for t in range(STEPS):
        if t < ANNEAL_START:
            lt = 0.0
        elif t < ANNEAL_START + ANNEAL_LEN:
            lt = lam * (t - ANNEAL_START + 1) / ANNEAL_LEN
        else:
            lt = float(lam)
        opt.zero_grad()
        for X, y in envs:
            z = net(X).squeeze(1)
            sig = torch.sigmoid(z)
            g_e = torch.mean((sig - y) * z)
            grad = (sig - y) / len(y) + lt * 2 * g_e.detach() * (sig * (1 - sig) * z + sig - y) / len(y)
            z.backward(grad / len(envs))
        opt.step()
    return net


@torch.no_grad()
def worst_acc_net(net, tests):
    net.eval()
    return min(((net(X).squeeze(1) > 0).float() == y).float().mean().item() for X, y in tests)


# ---------- official DomainBed baselines ----------
def make_hparams(algo, lam):
    hparams_registry, _ = _domainbed()
    hp = dict(hparams_registry.default_hparams(algo, 'ColoredMNIST'))
    hp.setdefault('mlp_dropout', 0.0)
    hp['mlp_width'], hp['mlp_depth'] = 128, 2
    hp['lr'] = LR
    if algo != 'ERM':
        hp[LAM_KEY[algo]] = float(lam)
        for k in list(hp):
            if k.endswith('penalty_anneal_iters'):
                hp[k] = ANNEAL_START
    return hp


def train_official(algo, hp, envs, seed):
    _, algorithms = _domainbed()
    torch.manual_seed(seed)
    np.random.seed(seed)
    alg = getattr(algorithms, algo)((392,), 2, len(envs), hp)
    alg.train()
    for _ in range(STEPS):
        alg.update([(X, y.long()) for X, y in envs])
    return alg


@torch.no_grad()
def worst_acc_alg(alg, tests):
    alg.eval()
    return min((alg.predict(X).argmax(1).float() == y).float().mean().item() for X, y in tests)


# ---------- subprocess worker: single job, prints one RESULT line ----------
def run_one(stage, gap, lam, seed):
    _load_mnist()
    envs, tests = build(gap, seed)
    gc.collect(); m0 = peak_mb(); t0 = time.time()       # data construction not timed
    if stage == 'r2':
        val = diagnostic_r2(envs)
    elif stage == 'OURS':
        val = worst_acc_net(train_ours(envs, lam, seed), tests)
    else:
        val = worst_acc_alg(train_official(stage, make_hparams(stage, lam), envs, seed), tests)
    dt = time.time() - t0
    mem = max(0.0, peak_mb() - m0)
    print(f'RESULT,{val:.6g},{dt:.2f},{mem:.0f}')


def done_set():
    if not os.path.exists(CSV):
        return set()
    out = set()
    for l in open(CSV):
        p = l.strip().split(',')
        if len(p) >= 4:
            out.add((p[0], p[1], f'{float(p[2]):g}', p[3]))  # lambda normalised to :g format
    return out


def run_job(script, done, stage, gap, lam, seed, note=''):
    key = (stage, f'{gap:g}', f'{lam:g}', str(seed))
    if key in done:
        return None
    proc = subprocess.run([sys.executable, script, stage, f'{gap:g}', f'{lam:g}', str(seed)],
                          capture_output=True, text=True)
    res = [l for l in proc.stdout.splitlines() if l.startswith('RESULT,')]
    if not res:
        print(f'FAILED {key}:\n{proc.stdout[-300:]}\n{proc.stderr[-500:]}', flush=True)
        return None
    val, dt, mem = res[0].split(',')[1:]
    with open(CSV, 'a') as f:
        f.write(f'{stage},{gap:g},{lam:g},{seed},{val},{dt}s,{mem}MB,{note}\n')
    print(f'[{stage}] gap={gap:g} lam={lam:g} seed={seed}: {float(val):.4f} '
          f'({dt}s, +{mem}MB) {note}', flush=True)
    return float(val)


def aggregate(gaps):
    """Write the paper-layout table: per gap, BIRD (diag + deployments) vs the
    best grid point of each baseline (grid time/memory = totals over all runs)."""
    rows = []
    if os.path.exists(CSV):
        rows = [l.strip().split(',') for l in open(CSV) if l.strip()]
    if not rows:
        print('no rows to aggregate (all jobs failed or nothing ran)')
        return
    by = {}
    for stage, gap, lam, seed, val, dt, mem, note in rows:
        by.setdefault((stage, float(gap), float(lam)), []).append(
            (float(val), float(dt.rstrip('s')), float(mem.rstrip('MB'))))
    header = ('r2_hat,BIRD_acc,ERM_acc,IRM_acc,VREx_acc,IB_IRM_acc,'
              'BIRD_time_s,ERM_time_s,IRM_time_s,VREx_time_s,IB_IRM_time_s,'
              'BIRD_mem_MBxruns,ERM_mem_MBxruns,IRM_mem_MBxruns,VREx_mem_MBxruns,IB_IRM_mem_MBxruns')
    lines = [header]
    for gap in sorted(gaps):
        r2 = by[('r2', gap, 0.0)][0][0]
        dep = [v for (s, g, l), v in by.items() if s == 'OURS' and g == gap]
        bird_acc = np.mean([x[0] for v in dep for x in v])
        diag_tm = by[('r2', gap, 0.0)][0]
        bird_time = diag_tm[1] + sum(x[1] for v in dep for x in v)
        bird_mem = diag_tm[2] + sum(x[2] for v in dep for x in v)
        accs, times, mems = [], [], []
        for algo in ['ERM'] + ALGOS:
            lams = [0.0] if algo == 'ERM' else LAM_GRID
            best, tsum, msum = -1.0, 0.0, 0.0
            for lam in lams:
                v = by.get((algo, gap, lam), [])
                if not v:
                    continue
                best = max(best, np.mean([x[0] for x in v]))
                tsum += sum(x[1] for x in v)
                msum += sum(x[2] for x in v)
            accs.append(best); times.append(tsum); mems.append(msum)
        lines.append(f"{r2:.4f},{bird_acc:.4f}," + ','.join(f'{a:.4f}' for a in accs) +
                     f",{bird_time:.1f}," + ','.join(f'{t:.1f}' for t in times) +
                     f",{bird_mem:.1f}," + ','.join(f'{m:.1f}' for m in mems))
    with open(TABLE, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n=== aggregated table (paper layout) ===')
    print('\n'.join(lines))
    print(f'\nRaw rows: {CSV}\nAggregated: {TABLE}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data-dir', default=DATA_DIR,
                    help='MNIST directory (default: <repo>/data or $BIRD_DATA_DIR)')
    ap.add_argument('--output-dir', default=OUT_DIR,
                    help='output directory (default: <repo>/results_generated or $BIRD_OUT_DIR)')
    ap.add_argument('--smoke', action='store_true',
                    help='quick check: gap=0.20, lambda in {100, 1000}, seed 0 (~5 min)')
    ap.add_argument('--no-baselines', action='store_true',
                    help='BIRD rows only (r2 diagnosis + deployments); no DomainBed needed')
    a = ap.parse_args()
    os.environ['BIRD_DATA_DIR'] = a.data_dir      # inherited by workers
    os.environ['BIRD_OUT_DIR'] = a.output_dir
    os.makedirs(a.output_dir, exist_ok=True)
    script = os.path.abspath(__file__)

    gaps = [0.20] if a.smoke else sorted(GAPS)
    lams = [100.0, 1000.0] if a.smoke else LAM_GRID
    seeds = [0] if a.smoke else SEEDS
    done = done_set()

    # ===== stage 1: BIRD (diagnosis -> anchored calibration -> deployment) =====
    print('=== stage 1: BIRD (one-off diagnosis route) ===', flush=True)
    r2 = {}
    for gap in gaps:
        v = run_job(script, done, 'r2', gap, 0.0, 0, 'diagnosis')
        if v is None:  # already done (or failed): recompute cheaply in-process
            _load_mnist()
            envs, _ = build(gap, 0)
            v = diagnostic_r2(envs)
        r2[gap] = v
    c0 = LAMBDA_CRIT_MEASURED[ANCHOR_GAP] * r2[ANCHOR_GAP]
    print(f'c0 = {LAMBDA_CRIT_MEASURED[ANCHOR_GAP]} x {r2[ANCHOR_GAP]:.4f} = {c0:.1f} '
          f'(anchor self-check: r^2(0.20) = 51.3874, c0 = 51387.4)', flush=True)
    for gap in gaps:
        lam_dep = 2 * c0 / r2[gap]
        for s in seeds:
            run_job(script, done, 'OURS', gap, lam_dep, s, 'deployment at 2*lambda*')

    # ===== stage 2: official DomainBed baselines on the lambda grid =====
    if not a.no_baselines:
        print('=== stage 2: DomainBed official baselines, lambda grid ===', flush=True)
        for gap in gaps:
            for algo in ['ERM'] + ALGOS:
                lam_list = [0.0] if algo == 'ERM' else lams
                for lam in lam_list:
                    for s in seeds:
                        run_job(script, done, algo, gap, lam, s)

    aggregate(gaps)


if __name__ == '__main__':
    if len(sys.argv) == 5:
        run_one(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))
    else:
        main()
