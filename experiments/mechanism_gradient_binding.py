# -*- coding: utf-8 -*-
"""
Mechanism probe (companion to Section 4.2.5): why does accuracy degrade after
nearest-neighbour upsampling? (56x56 testbed)
==============================================================================
Background: with the deployment recipe carried over verbatim, BIRD worst-env
accuracy was 0.6348/0.6540 at 14x14 but 0.4108/0.3888 at 140x140. The
diagnosis side is provably resolution-invariant (r_hat^2 = 0.001338 and
lambda = 2000 at both scales), so the degradation must be on the deployment
side.

Hypothesis H (gradient binding of Adam x nearest-neighbour replication):
nearest-neighbour upsampling replicates each pixel into a block of
k = (side/14)^2 pixels. All pixels in a block take identical values on EVERY
sample -> identical first-layer weight gradients -> Adam's per-coordinate
normalisation (a +-lr sign-like update each step) moves the tied weights in
lockstep -> the function-space step of the block's effective weight (the sum
of the k tied weights) is amplified k-fold. At 140x140, k = 100: with the same
lr = 1e-3 and 400 steps, function-space progress is ~100x that of 14x14, so
the spurious feature is deeply overfit before annealing (t = 150) starts and
the penalty cannot recover it.

Testable predictions:
  P1 binding evidence: within-block first-layer weight gradients are bitwise
     equal; after one Adam step, within-block weight increments are bitwise equal
  P2 repair prediction: side=56 (k=16) with lr = 1e-3/16 = 6.25e-5 should
     return acc to ~0.64 (the 14x14 level)  [NOTE: this UNIFORM-lr repair was
     FALSIFIED here; the correct layered-lr repair is mechanism_layered_lr.py]
  P3 dose-response: lr in {1e-3, 2.5e-4, 6.25e-5, 1.56e-5} at side 56 should
     peak at 6.25e-5 (if so, 6.25e-5 is derived a priori, not searched --
     both neighbours are worse)
  P4 trajectory evidence: at lr = 1e-3 the pre-anneal (t <= 149) worst-acc has
     already collapsed to the spurious level; at lr/k the 56x56 trajectory
     should coincide step-by-step with the 14x14 trajectory

Protocol identical to fig8_scalability.py (build/make_env/diag_r2/train_net
copied verbatim; the only change: train_net gains an lr parameter and
trajectory recording). The side=14, lr=1e-3 anchor should reproduce
0.6328~0.6540. All random call order unchanged.

Usage:
  python experiments/mechanism_gradient_binding.py                 # full probe (~15 min CPU)
  python experiments/mechanism_gradient_binding.py --data-dir D
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
    net.eval()
    a = min(((net(X).squeeze(1) > 0).float() == y).float().mean().item() for X, y in tests)
    net.train()
    return a


CHECKPOINTS = (0, 25, 50, 100, 149, 200, 250, 300, 399)


def train_net(envs, lam, seed, d, lr=LR, tests=None):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    traj = {}
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
        if tests is not None and t in CHECKPOINTS:
            traj[t] = worst_acc_net(net, tests)
    return net, traj


def tie_check(side):
    """P1: are within-block inputs, gradients and Adam increments bitwise tied?"""
    s = side // 14
    envs, _ = build(side, 0)
    X, y = envs[0]
    d = 2 * side * side
    # input binding: within-block max-min over channel 0
    Xc = X[:, :side * side].reshape(-1, side, side)
    Xb = Xc.reshape(len(Xc), 14, s, 14, s)
    in_tie = (Xb.amax(dim=(2, 4)) - Xb.amin(dim=(2, 4))).abs().max().item()
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 1))
    loss = F.binary_cross_entropy_with_logits(net(X).squeeze(1), y)
    loss.backward()
    g = net[0].weight.grad[:, :side * side].reshape(128, 14, s, 14, s)
    g_tie = (g - g.mean(dim=(2, 4), keepdim=True)).abs().max().item()
    w0 = net[0].weight.detach().clone()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    opt.step()
    dw = (net[0].weight.detach() - w0)[:, :side * side].reshape(128, 14, s, 14, s)
    dw_tie = (dw - dw.mean(dim=(2, 4), keepdim=True)).abs().max().item()
    print(f'[P1 tie check side={side}, k={s*s}] within-block max|dx| = {in_tie:.3e} | '
          f'gradient max|dg| = {g_tie:.3e} | Adam increment max|dw| = {dw_tie:.3e}', flush=True)


def run(side, lr, seed):
    envs, tests = build(side, seed)
    envs0, _ = build(side, 0)
    lam = 2 * C0 / diag_r2(envs0)
    del envs0
    t0 = time.time()
    net, traj = train_net(envs, lam, seed, 2 * side * side, lr=lr, tests=tests)
    acc = worst_acc_net(net, tests)
    ts = ' '.join(f'{t}:{v:.3f}' for t, v in traj.items())
    print(f'[run] side={side} k={(side//14)**2} lr={lr:g} seed={seed} lam={lam:.4g} -> '
          f'worst-acc={acc:.4f} ({time.time()-t0:.0f}s)\n      traj[{ts}]', flush=True)
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data-dir', default=DATA_DIR,
                    help='MNIST directory (default: <repo>/data or $BIRD_DATA_DIR)')
    a = ap.parse_args()
    os.environ['BIRD_DATA_DIR'] = a.data_dir
    _load_mnist()

    print('=== mechanism probe: gradient binding of Adam x nearest-neighbour replication ===', flush=True)
    tie_check(56)
    # anchor: 14x14 original protocol (should reproduce ~0.6328/0.6536)
    run(14, 1e-3, 0)
    # P4 + baseline reproduction: 56x56 at the original lr (previously measured 0.4400/0.5016)
    run(56, 1e-3, 0)
    run(56, 1e-3, 1)
    # P2 repair prediction: lr = 1e-3/k = 6.25e-5
    run(56, 6.25e-5, 0)
    run(56, 6.25e-5, 1)
    # P3 dose-response bracket: 4x overshoot / 4x undershoot
    run(56, 2.5e-4, 0)
    run(56, 1.5625e-5, 0)
    # cross-scale invariance: side=28, k=4, lr=2.5e-4 should also return to ~0.64
    run(28, 2.5e-4, 0)
    run(28, 2.5e-4, 1)
    print('=== done ===', flush=True)


if __name__ == '__main__':
    main()
