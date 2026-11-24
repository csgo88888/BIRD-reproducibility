# -*- coding: utf-8 -*-
"""
Ablation study (paper Section 4.2.4, Figure 7): accuracy / time / memory
=========================================================================
Same protocol as Section 4.2.1 (Figure 5): ColoredMNIST 14x14, N=5000 per
environment, gaps Delta in {0.10, 0.20, 0.30}, anchor constant
C0 = 1000 * 51.3874 (lambda_crit = 1000 measured at Delta = 0.20), diagnosis
fixed at seed 0, deployment seeds {0, 1}.

Variants (one component removed at a time):
  full        full BIRD: label-aligned means -> centring -> r^2 -> lambda = 2*C0/r^2
  no_align    without label alignment: unsigned means (r^2 collapses -> lambda explodes)
  no_center   without centring: scatter = sum m_e^2 (r^2 inflated -> lambda too weak)
  no_scalelaw without the scaling law: lambda fixed at 2000 (anchor value)
  no_safety   without the safety factor: lambda = C0/r^2

Usage:
  python experiments/fig7_ablation.py                 # full run (45 subprocess jobs)
  python experiments/fig7_ablation.py --smoke         # smoke test: full variant, Delta=0.20, seed 0
  python experiments/fig7_ablation.py --data-dir D --output-dir O

Options:
  --data-dir    directory holding the MNIST raw files (downloaded automatically
                on first run; default <repo>/data)
  --output-dir  where fig7_ablation_raw.csv is written (default <repo>/results_generated)

Memory protocol: every job runs in a separate subprocess so the peak RSS of a
single run is measured cleanly (peak RSS is a high-water mark and would be
reused within one process), identical to the Figure 5 protocol.
Checkpoint resume: completed (variant, gap, seed, role) rows are skipped, so a
finished run can be re-run safely and an interrupted run continues where it
stopped. Expected runtime: ~20-30 min on a CPU (each job ~10-20 s).
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GAPS = {0.10: (0.8, 0.9), 0.20: (0.75, 0.95), 0.30: (0.7, 1.0)}
N_PER_ENV, STEPS, LR = 5000, 400, 1e-3
ANNEAL_START, ANNEAL_LEN = 150, 100
C0 = 1000.0 * 51.3874                      # anchor (gap=0.20, lambda_crit=1000)
SEEDS = [0, 1]
VARIANTS = ['full', 'no_align', 'no_center', 'no_scalelaw', 'no_safety']

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get('BIRD_DATA_DIR') or os.path.join(REPO_ROOT, 'data')
OUT_DIR = os.environ.get('BIRD_OUT_DIR') or os.path.join(REPO_ROOT, 'results_generated')
CSV = os.path.join(OUT_DIR, 'fig7_ablation_raw.csv')

TR_X = TR_Y = TE_X = TE_Y = None  # lazily loaded (only in worker subprocesses)


def _load_mnist():
    global TR_X, TR_Y, TE_X, TE_Y
    if TR_X is None:
        from torchvision.datasets import MNIST
        tr = MNIST(DATA_DIR, train=True, download=True)
        te = MNIST(DATA_DIR, train=False, download=True)
        TR_X, TR_Y = tr.data.numpy(), tr.targets.numpy()
        TE_X, TE_Y = te.data.numpy(), te.targets.numpy()


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


def diag_r2(envs, variant):
    zy = [((X[:, :196] - X[:, 196:]).sum(1).numpy(), y.numpy()) for X, y in envs]
    if variant == 'no_align':
        m_e = np.array([z.mean() for z, y in zy])
    else:
        m_e = np.array([(z * (2 * y - 1)).mean() for z, y in zy])
    if variant == 'no_center':
        return float((m_e ** 2).sum())
    return float(((m_e - m_e.mean()) ** 2).sum())


def lam_of(variant, r2):
    if variant == 'no_scalelaw':
        return 2000.0
    if variant == 'no_safety':
        return C0 / r2
    return 2 * C0 / r2


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


# ---------- subprocess worker: single job, prints one RESULT line ----------
def run_one(variant, gap, seed, role):
    import gc
    _load_mnist()
    if role == 'diag':
        envs, _ = build(gap, 0)
        gc.collect(); m0 = peak_mb(); t0 = time.time()   # data construction not timed
        acc, lam = diag_r2(envs, variant), 0.0
    else:
        envs0, _ = build(gap, 0)
        lam = lam_of(variant, diag_r2(envs0, variant))
        envs, tests = build(gap, seed)
        gc.collect(); m0 = peak_mb(); t0 = time.time()
        acc = worst_acc_net(train_ours(envs, lam, seed), tests)
    dt = time.time() - t0
    mem = max(0.0, peak_mb() - m0)
    print(f'RESULT,{acc:.4f},{dt:.2f},{mem:.0f},{lam:.6g}')


def done_set():
    if not os.path.exists(CSV):
        return set()
    return {tuple(l.strip().split(',')[:4]) for l in open(CSV) if l.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data-dir', default=DATA_DIR,
                    help='MNIST directory (default: <repo>/data or $BIRD_DATA_DIR)')
    ap.add_argument('--output-dir', default=OUT_DIR,
                    help='output directory (default: <repo>/results_generated or $BIRD_OUT_DIR)')
    ap.add_argument('--smoke', action='store_true',
                    help='quick check: full variant, gap=0.20, seed 0 only (~30 s)')
    a = ap.parse_args()
    os.environ['BIRD_DATA_DIR'] = a.data_dir      # inherited by workers
    os.environ['BIRD_OUT_DIR'] = a.output_dir
    os.makedirs(a.output_dir, exist_ok=True)
    script = os.path.abspath(__file__)

    variants = ['full'] if a.smoke else VARIANTS
    gaps = [0.20] if a.smoke else sorted(GAPS)
    seeds = [0] if a.smoke else SEEDS

    done = done_set()
    for variant in variants:
        for gap in gaps:
            jobs = [('diag', 0)] + [('deploy', s) for s in seeds]
            for role, seed in jobs:
                key = (variant, f'{gap:g}', str(seed), role)
                if key in done:
                    continue
                proc = subprocess.run(
                    [sys.executable, script, variant, f'{gap:g}', str(seed), role],
                    capture_output=True, text=True)
                res = [l for l in proc.stdout.splitlines() if l.startswith('RESULT,')]
                if not res:
                    print(f'FAILED {key}:\n{proc.stdout[-300:]}\n{proc.stderr[-500:]}', flush=True)
                    continue
                acc, dt, mem, lam = res[0].split(',')[1:]
                with open(CSV, 'a') as f:
                    f.write(f'{variant},{gap:g},{seed},{role},{acc},{dt}s,{mem}MB,{lam}\n')
                print(f'[{variant}] gap={gap} seed={seed} {role}: {float(acc):.4f} '
                      f'({dt}s, +{mem}MB, lam={float(lam):g})', flush=True)

    # ---------- summary (same aggregation as Figure 5: acc = seed mean;
    # time = diagnosis + deployments; memory = sum of per-run peaks) ----------
    if not os.path.exists(CSV):
        return
    rows = [l.strip().split(',') for l in open(CSV) if l.strip()]
    agg = {}
    for variant, gap, seed, role, acc, dt, mem, lam in rows:
        k = (variant, float(gap))
        a = agg.setdefault(k, {'acc': [], 'time': 0.0, 'mem': 0.0})
        if role == 'deploy':
            a['acc'].append(float(acc))
        a['time'] += float(dt.rstrip('s'))
        a['mem'] += float(mem.rstrip('MB'))
    print('\n=== summary (for plotting) ===')
    print('variant,gap,acc_mean,time_s,mem_MBxruns')
    for (variant, gap), a in sorted(agg.items()):
        acc = f"{np.mean(a['acc']):.4f}" if a['acc'] else 'NA'
        print(f"{variant},{gap:g},{acc},{a['time']:.1f},{a['mem']:.0f}")
    print(f'\nRaw rows: {CSV}')
    print('Anchor self-check: full/diag at gap=0.20 should give r^2 = 51.3874 '
          '(lambda = 2000); full/deploy seed 0/1 should give worst-env acc ~ 0.6328 / 0.6536.')


if __name__ == '__main__':
    if len(sys.argv) == 5:
        run_one(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else:
        main()
