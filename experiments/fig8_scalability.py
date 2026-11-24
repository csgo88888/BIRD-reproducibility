# -*- coding: utf-8 -*-
"""
Scalability study (paper Section 4.2.5, Figure 8): accuracy / time / memory
============================================================================
Same protocol as Section 4.2.1 (Figure 5), the only change being the input
resolution: ColoredMNIST side in {14, 140} (d = 2*side^2 in {392, 39200}),
fixed gap Delta = 0.20 (anchor condition), N=5000 per environment, 400 steps,
Adam lr = 1e-3, MLP (2*side^2) -> 128 -> 1.
Data generation: build the 14x14 task exactly as in fig7_ablation.py, then
nearest-neighbour upsample to side x side -- same task, more pixels, no extra
information, only more computation. At side=14 the data are byte-identical to
fig7_ablation.py (reproduces r^2_norm = 0.001338, lambda = 2000, BIRD
seed 0/1 acc ~ 0.6328/0.6536).

side = 1400 (d = 3.92M) is not feasible on a single machine (float32 data
~ 235 GB, ~33 h per CPU training run). That point is NOT measured; it is
projected by plotting/plot_fig8.py under the disclosed linear-in-d cost model
(dashed line + hollow marker, accuracy panel not projected).

Methods:
  bird  full BIRD: diagnosis r^2 -> lambda = 2*C0/r^2 -> deployment
        (lambda frozen from the seed-0 diagnosis, per protocol)
  erm   official DomainBed ERM (same convention as Figure 5: default_hparams +
        width 128 / depth 2 / lr 1e-3; anchor reproduces ~ 0.2932/0.3000).
        Note: a hand-rolled lambda=0 loop is NOT the official ERM baseline.

Resolution invariance of the diagnosis: the residual probe is a per-pixel
*mean* (unit-norm direction), so r^2_norm = 51.3874/196^2 ~= 0.001338 is
exactly invariant under nearest-neighbour upsampling and lambda = 2000 at
every scale (C0 converted accordingly).

Deployment at higher resolutions uses a *layered* learning rate: the input
layer uses lr/k with k = (side/14)^2, all other parameters keep lr. Reason
(verified bitwise by experiments/mechanism_gradient_binding.py and
mechanism_layered_lr.py): nearest-neighbour upsampling creates blocks of
k pixels whose gradients are tied exactly; Adam's per-coordinate +-lr update
amplifies the function-space step of the block's effective weight by k, while
the readout layer is unchanged -- the same recipe at high resolution is
equivalent to 14x14 training with input-layer lr*k and overshoots the
spurious feature before annealing. At side=14 (k=1) the original
single-parameter-group path is used, byte-identical to fig5/fig7.

Usage:
  python experiments/fig8_scalability.py                # full run (~2-2.5 h CPU)
  python experiments/fig8_scalability.py --smoke        # side=14, BIRD only, seed 0 (~1 min)
  python experiments/fig8_scalability.py --data-dir D --output-dir O

The ERM baseline requires DomainBed (official classes, same source as
Figure 5):
  git clone --depth 1 https://github.com/facebookresearch/DomainBed.git
Place it in the repository root or point the environment variable
DOMAINBED_PATH at it. BIRD-only runs do not need DomainBed.

Memory protocol: this experiment reports the *absolute* process peak RSS
(MB), not the relative increment of Section 4.2.4 -- a scalability study
cares about the total pipeline footprint (the dataset itself is the dominant
term at large side). Every job still runs in a separate subprocess.
Checkpoint resume: completed (side, method, seed, role) rows are skipped.
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

P1, P2 = 0.75, 0.95                            # gap Delta = 0.20 (anchor condition)
N_PER_ENV, STEPS, LR = 5000, 400, 1e-3
ANNEAL_START, ANNEAL_LEN = 150, 100
# Residual probe = unit-norm direction (per-pixel mean, not sum).
# r^2_norm = 51.3874/196^2 ~= 0.001338 is exactly invariant under nearest-
# neighbour upsampling, so lambda = 2000 at every scale; deployment at
# side=14 is identical to fig5/fig7 (same lambda, same training).
C0 = 1000.0 * 51.3874 / 196**2                 # anchor (gap=0.20, lambda_crit=1000), normalised units
SEEDS = [0, 1]
SIDES = [14, 140]                              # measured scales; 1400 is projected by the plot script
METHODS = ['bird', 'erm']

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get('BIRD_DATA_DIR') or os.path.join(REPO_ROOT, 'data')
OUT_DIR = os.environ.get('BIRD_OUT_DIR') or os.path.join(REPO_ROOT, 'results_generated')
CSV = os.path.join(OUT_DIR, 'fig8_scalability_raw.csv')

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
    """Lazy import of DomainBed (only needed for the official ERM baseline)."""
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
                '(also requires `pip install timm`)')
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


def make_env(images, digits, p_color, rng, side):
    # RNG call order identical to fig7_ablation.py; at side=14 the data are
    # byte-identical to fig7_ablation.py
    y = (digits < 5).astype(np.float32)
    y = np.where(rng.random(len(y)) < 0.25, 1 - y, y)
    color = np.where(rng.random(len(y)) < p_color, y, 1 - y)
    img = F.avg_pool2d(torch.from_numpy(images).float().unsqueeze(1) / 255., 2).squeeze(1)
    if side != 14:
        img = F.interpolate(img.unsqueeze(1), scale_factor=side // 14, mode='nearest').squeeze(1)
    X = torch.zeros(len(y), 2, side, side)
    c = torch.from_numpy(color)
    X[:, 0] = img * c[:, None, None]
    X[:, 1] = img * (1 - c)[:, None, None]
    return X.reshape(len(y), -1), torch.from_numpy(y)


def build(side, seed):
    rng = np.random.default_rng(123)
    perm = rng.permutation(60000)
    te_perm = rng.permutation(10000)
    erng = np.random.default_rng(1000 * seed + 20)       # gap=0.20 -> +20, as in fig7
    envs = [make_env(TR_X[perm[:N_PER_ENV]], TR_Y[perm[:N_PER_ENV]], P1, erng, side),
            make_env(TR_X[perm[N_PER_ENV:2*N_PER_ENV]], TR_Y[perm[N_PER_ENV:2*N_PER_ENV]], P2, erng, side)]
    tests = [make_env(TE_X[te_perm[:2500]], TE_Y[te_perm[:2500]], 0.1, erng, side),
             make_env(TE_X[te_perm[2500:5000]], TE_Y[te_perm[2500:5000]], 0.9, erng, side)]
    return envs, tests


def diag_r2(envs):
    npx = envs[0][0].shape[1] // 2
    zy = [((X[:, :npx] - X[:, npx:]).mean(1).numpy(), y.numpy()) for X, y in envs]
    m_e = np.array([(z * (2 * y - 1)).mean() for z, y in zy])
    return float(((m_e - m_e.mean()) ** 2).sum())


def train_net(envs, lam, seed, d, side=14):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 1))
    k = (side // 14) ** 2                            # area of a nearest-neighbour tied block (140 -> 100)
    if k == 1:
        opt = torch.optim.Adam(net.parameters(), lr=LR)      # 14x14: original protocol unchanged
    else:
        # layered lr: tied blocks amplify the input layer's function-space step by k; divide it back.
        # Readout layer and biases are not tied and keep lr.
        opt = torch.optim.Adam([
            {'params': [net[0].weight], 'lr': LR / k},
            {'params': [net[0].bias, net[2].weight, net[2].bias], 'lr': LR},
        ], lr=LR)
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


def train_erm(envs, seed, d):
    # Identical to fig5/Figure 5: official DomainBed ERM class + default
    # hyperparameters (width 128 / depth 2 / lr 1e-3)
    hparams_registry, algorithms = _domainbed()
    torch.manual_seed(seed)
    np.random.seed(seed)
    hp = dict(hparams_registry.default_hparams('ERM', 'ColoredMNIST'))
    hp.setdefault('mlp_dropout', 0.0)
    hp['mlp_width'], hp['mlp_depth'] = 128, 2
    hp['lr'] = LR
    alg = algorithms.ERM((d,), 2, len(envs), hp)
    alg.train()
    for _ in range(STEPS):
        alg.update([(X, y.long()) for X, y in envs])
    return alg


@torch.no_grad()
def worst_acc_alg(alg, tests):
    alg.eval()
    return min((alg.predict(X).argmax(1).float() == y).float().mean().item() for X, y in tests)


@torch.no_grad()
def worst_acc_net(net, tests):
    net.eval()
    return min(((net(X).squeeze(1) > 0).float() == y).float().mean().item() for X, y in tests)


# ---------- subprocess worker: single job, prints one RESULT line ----------
def run_one(side, method, seed, role):
    import gc
    _load_mnist()
    if role == 'diag':
        envs, _ = build(side, 0)
        gc.collect(); t0 = time.time()                   # data construction not timed
        acc, lam = diag_r2(envs), 0.0
    else:
        envs, tests = build(side, seed)
        if method == 'bird':
            envs0, _ = build(side, 0)
            lam = 2 * C0 / diag_r2(envs0)
            del envs0
            gc.collect(); t0 = time.time()
            acc = worst_acc_net(train_net(envs, lam, seed, 2 * side * side, side), tests)
        else:
            lam = 0.0
            gc.collect(); t0 = time.time()
            acc = worst_acc_alg(train_erm(envs, seed, 2 * side * side), tests)
    dt = time.time() - t0
    mem = peak_mb()                                      # absolute peak (this study's convention)
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
                    help='quick check: side=14, BIRD only, seed 0 (~1 min)')
    a = ap.parse_args()
    os.environ['BIRD_DATA_DIR'] = a.data_dir
    os.environ['BIRD_OUT_DIR'] = a.output_dir
    os.makedirs(a.output_dir, exist_ok=True)
    script = os.path.abspath(__file__)

    sides = [14] if a.smoke else SIDES
    methods = ['bird'] if a.smoke else METHODS
    seeds = [0] if a.smoke else SEEDS

    done = done_set()
    for side in sides:
        for method in methods:
            jobs = ([('diag', 0)] if method == 'bird' else []) + [('deploy', s) for s in seeds]
            for role, seed in jobs:
                key = (str(side), method, str(seed), role)
                if key in done:
                    continue
                proc = subprocess.run(
                    [sys.executable, script, str(side), method, str(seed), role],
                    capture_output=True, text=True)
                res = [l for l in proc.stdout.splitlines() if l.startswith('RESULT,')]
                if not res:
                    print(f'FAILED {key}:\n{proc.stdout[-300:]}\n{proc.stderr[-500:]}', flush=True)
                    continue
                acc, dt, mem, lam = res[0].split(',')[1:]
                with open(CSV, 'a') as f:
                    f.write(f'{side},{method},{seed},{role},{acc},{dt}s,{mem}MB,{lam}\n')
                print(f'[side={side}] {method} seed={seed} {role}: {float(acc):.4f} '
                      f'({dt}s, peak {mem}MB, lam={float(lam):g})', flush=True)

    # ---------- summary (for plotting) ----------
    if not os.path.exists(CSV):
        return
    rows = [l.strip().split(',') for l in open(CSV) if l.strip()]
    agg = {}
    for side, method, seed, role, acc, dt, mem, lam in rows:
        k = (method, int(side))
        a = agg.setdefault(k, {'acc': [], 'time': 0.0, 'mem': 0.0})
        if role == 'deploy':
            a['acc'].append(float(acc))
        a['time'] += float(dt.rstrip('s'))
        a['mem'] += float(mem.rstrip('MB'))
    print('\n=== summary (for plotting) ===')
    print('method,side,acc_mean,time_s,mem_MBxruns')
    for (method, side), a in sorted(agg.items()):
        acc = f"{np.mean(a['acc']):.4f}" if a['acc'] else 'NA'
        print(f"{method},{side},{acc},{a['time']:.1f},{a['mem']:.0f}")
    # anchor self-check: side=14 should reproduce r^2_norm = 51.3874/196^2 ~= 0.001338,
    # lambda = 2000, bird acc ~ 0.6328/0.6536
    print('\n=== anchor self-check (side=14 must match Section 4.2.1/Figure 7) ===')
    for side, method, seed, role, acc, dt, mem, lam in rows:
        if int(side) == 14:
            print(f'{method} seed={seed} {role}: acc/r2={acc}, lam={lam}')
    print(f'\nRaw rows: {CSV}')


if __name__ == '__main__':
    if len(sys.argv) == 5:
        run_one(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        main()
