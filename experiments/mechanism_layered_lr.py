# -*- coding: utf-8 -*-
"""
Mechanism verdict (companion to Section 4.2.5): layered-learning-rate
equivalence test (follows mechanism_gradient_binding.py)
==============================================================================
Findings from mechanism_gradient_binding.py: the binding structure exists
bitwise (P1), but the uniform lr/k repair FAILS (P2 falsified) -- the anneal
recovery phase at 56x56 is unstable/stagnant with strong seed variance.

Exact equivalence argument: inside the tied subspace the function depends only
on the block-sum weight W = sum_{i in block} w_i. Adam's per-coordinate +-lr
update gives W a function-space speed of k*lr, while the untied parameters
(w2, b1, b2) move at speed lr. Hence side x side tied training is EQUIVALENT
to 14x14 training in which the first layer has lr_eff = k*lr and all other
parameters keep lr. The correct repair is therefore not uniform lr/k but a
LAYERED rate: W1 <- lr/k, w2/b1/b2 <- lr unchanged. Under this parameterisation
56x56 should coincide with 14x14 checkpoint by checkpoint, ending at
~0.6328/0.6536.

This script:
  T1: side=56, param groups {W1: 1e-3/16, others: 1e-3}, seeds 0/1 -> expect ~0.63
  T2: same runs record binding drift (within-block weight std at t=0/150/399)
      -> checks whether floating-point noise breaks the tie
  T3: side=28, param groups {W1: 1e-3/4, others: 1e-3}, seeds 0/1 -> cross-validation
Controls (already established by mechanism_gradient_binding.py): 14x14 original
protocol 0.6340; 56x56 uniform lr in {1e-3, 2.5e-4, 6.25e-5, 1.56e-5} ->
0.440/0.502/0.449/0.461/0.502/0.571 (none recovers).

This layered rate is exactly what fig8_scalability.py uses for side > 14.

Usage:
  python experiments/mechanism_layered_lr.py                 # (~10 min CPU)
  python experiments/mechanism_layered_lr.py --data-dir D
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

P1_, P2_ = 0.75, 0.95
N_PER_ENV, STEPS, LR = 5000, 400, 1e-3
ANNEAL_START, ANNEAL_LEN = 150, 100
C0 = 1000.0 * 51.3874 / 196**2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get('BIRD_DATA_DIR') or os.path.join(REPO_ROOT, 'data')

TR_X = TR_Y = TE_X = TE_Y = None  # lazily loaded


def _load_mnist():
    global TR_X, TR_Y, TE_X, TE_Y
    if TR_X is None:
        from torchvision.datasets import MNIST
        tr = MNIST(DATA_DIR, train=True, download=True)
        te = MNIST(DATA_DIR, train=False, download=True)
        TR_X, TR_Y = tr.data.numpy(), tr.targets.numpy()
        TE_X, TE_Y = te.data.numpy(), te.targets.numpy()


def make_env(images, digits, p_color, rng, side):
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
    erng = np.random.default_rng(1000 * seed + 20)
    envs = [make_env(TR_X[perm[:N_PER_ENV]], TR_Y[perm[:N_PER_ENV]], P1_, erng, side),
            make_env(TR_X[perm[N_PER_ENV:2*N_PER_ENV]], TR_Y[perm[N_PER_ENV:2*N_PER_ENV]], P2_, erng, side)]
    tests = [make_env(TE_X[te_perm[:2500]], TE_Y[te_perm[:2500]], 0.1, erng, side),
             make_env(TE_X[te_perm[2500:5000]], TE_Y[te_perm[2500:5000]], 0.9, erng, side)]
    return envs, tests


def diag_r2(envs):
    npx = envs[0][0].shape[1] // 2
    zy = [((X[:, :npx] - X[:, npx:]).mean(1).numpy(), y.numpy()) for X, y in envs]
    m_e = np.array([(z * (2 * y - 1)).mean() for z, y in zy])
    return float(((m_e - m_e.mean()) ** 2).sum())


@torch.no_grad()
def worst_acc_net(net, tests):
    return min(((net(X).squeeze(1) > 0).float() == y).float().mean().item() for X, y in tests)


CHECKPOINTS = (0, 25, 50, 100, 149, 200, 250, 300, 399)


def train_net_pg(envs, lam, seed, d, lr_w1, lr_rest, side, tests):
    """Layered lr: net[0].weight <- lr_w1; net[0].bias, net[2] <- lr_rest"""
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 1))
    opt = torch.optim.Adam([
        {'params': [net[0].weight], 'lr': lr_w1},
        {'params': [net[0].bias, net[2].weight, net[2].bias], 'lr': lr_rest},
    ], lr=lr_rest)
    s = side // 14
    traj, tie_drift = {}, {}
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
        if t in CHECKPOINTS:
            traj[t] = worst_acc_net(net, tests)
            W = net[0].weight.detach()[:, :side * side].reshape(128, 14, s, 14, s)
            tie_drift[t] = W.std(dim=(2, 4)).max().item()  # within-block weight spread (binding drift)
    return net, traj, tie_drift


def run_pg(side, seed):
    k = (side // 14) ** 2
    envs, tests = build(side, seed)
    envs0, _ = build(side, 0)
    lam = 2 * C0 / diag_r2(envs0)
    del envs0
    t0 = time.time()
    net, traj, tie = train_net_pg(envs, lam, seed, 2 * side * side,
                                  LR / k, LR, side, tests)
    acc = worst_acc_net(net, tests)
    ts = ' '.join(f'{t}:{v:.3f}' for t, v in traj.items())
    td = ' '.join(f'{t}:{v:.2e}' for t, v in tie.items())
    print(f'[pg] side={side} k={k} lr_w1={LR/k:.3g} lr_rest={LR:g} seed={seed} '
          f'lam={lam:.4g} -> worst-acc={acc:.4f} ({time.time()-t0:.0f}s)\n'
          f'     traj[{ts}]\n     tie_drift[{td}]', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data-dir', default=DATA_DIR,
                    help='MNIST directory (default: <repo>/data or $BIRD_DATA_DIR)')
    a = ap.parse_args()
    os.environ['BIRD_DATA_DIR'] = a.data_dir
    _load_mnist()

    print('=== mechanism verdict: layered learning rate ===', flush=True)
    run_pg(56, 0)
    run_pg(56, 1)
    run_pg(28, 0)
    run_pg(28, 1)
    print('=== done ===', flush=True)


if __name__ == '__main__':
    main()
