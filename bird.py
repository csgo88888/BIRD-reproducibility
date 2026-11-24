# -*- coding: utf-8 -*-
"""
BIRD (Blind IRM Residual Diagnostic) — reference implementation
================================================================
A faithful implementation of the three algorithms of Section 3:

  Algorithm 1 (Section 3.1, blind diagnosis)  -> `diagnose_blind`
      label-aligned environment mean vectors v_e = E[x*(2y-1)] over ALL input
      dimensions (no pre-selected probe), with the split-half cross estimator
      of Theorem 1 for an unbiased residual r_hat^2; the eigen-structure of
      the scatter matrix S = sum_e (v_e - v_bar)(v_e - v_bar)^T gives the
      spurious direction (largest eigenvalue) and the supplementation
      direction v_hat (smallest eigenvalue -- where environmental variation
      is nearly zero, cf. Figure 2 and Lemma A.1). Decision yes/no: the
      residual is estimable iff N_e * r_hat^2 >= 100 (the 20%-relative-error
      floor of Theorem 1, used as the anchor rule in Algorithm 2).

  Algorithm 2 (Section 3.2, one-anchor calibration) -> `calibrate_c0_search`
      select as anchor the candidate condition with the largest r_hat^2
      satisfying N_e * r_hat^2 >= 100; sweep the probe grid
      {1e2, 1e3, 1e4, 1e5}, record worst-group accuracy, and take the
      knee (plateau turning point, tolerance eps = 0.01) as lambda_crit;
      c0 = lambda_crit * r_hat^2(anchor). On the paper's ColoredMNIST
      benchmark this recovers lambda_crit = 1000, c0 = 51387.4.
      `calibrate_c0` remains for the direct closed form c0 = lambda_crit * r2.

  Algorithm 3 (Section 3.3, anchored deployment) -> `deploy`
      (r_hat^2, v_hat, decision) <- Algorithm 1; if decision = no, return
      ('no', v_hat) for environment supplementation; lambda = kappa * c0 /
      r_hat^2 with safety factor kappa = 2; if lambda > lambda_max (the
      trainability ceiling, kappa ~ 5.8 in the paper's experiments), return
      'no'; otherwise train ONCE with the annealed IRMv1 penalty.

The Section-4 experiments diagnose with the known channel-difference probe
(ColoredMNIST's spurious feature is the colour channel by construction);
`diagnose_r2` / `colored_mnist_probe` keep that convention. The blind variant
(Algorithm 1) is the split-half unbiased estimator of the same residual and
was validated on this benchmark. To *reproduce the paper figures*, run the
self-contained scripts under `experiments/`; this module is for *using* BIRD.

UNIT CONTRACT (important): c0 always carries the units of the diagnosis it
was calibrated with. The paper's constant c0 = 51387.4 is in PROBE units
(channel-difference, per-pixel SUM convention) and must only be combined with
`diagnose_r2` / `deploy(probe=...)`. A c0 returned by `calibrate_c0_search`
is in BLIND (raw-vector) units and must only be combined with
`deploy(probe=None)`. Mixing units makes lambda meaningless.
"""
import numpy as np
import torch
import torch.nn as nn


def _np(a):
    """Accept numpy arrays or torch tensors; return a numpy array."""
    if torch.is_tensor(a):
        return a.detach().cpu().numpy()
    return np.asarray(a)


# The paper's calibrated class constant on ColoredMNIST (Section 4):
# c0 = lambda_crit * r_hat^2 = 1000 * 51.3874, in PROBE units (channel-
# difference, per-pixel SUM convention). Use only with the probe path
# (diagnose_r2 / deploy(probe=...)); see the unit contract above.
C0_COLORED_MNIST = 51387.4


# ---------------------------------------------------------------------------
# Probe-based diagnosis (Section 4 convention: spurious direction known)
# ---------------------------------------------------------------------------
def label_aligned_means(envs, probe):
    """m_e = mean over samples of (probe . x) * (2y - 1), per environment.

    envs  : list of (X, y) with X of shape (n, d) and y in {0, 1}
    probe : direction in input space (numpy array of shape (d,)). For the
            ColoredMNIST experiments this is the channel-difference direction
            (see `colored_mnist_probe`). The label-alignment factor (2y - 1)
            removes the class-conditional sign flip (Section 3.1).
    """
    probe = np.asarray(probe, dtype=np.float64)
    return np.array([((_np(X) @ probe) * (2 * _np(y) - 1)).mean()
                     for X, y in envs])


def diagnose_r2(envs, probe, center=True):
    """Residual magnitude along a known probe: r_hat^2 = sum_e (m_e - m_bar)^2.

    Centring (the default) subtracts the grand mean so that only the
    *variation across environments* is measured; the ablation variant
    `no_center` (Section 4.2.4) corresponds to center=False.
    """
    m_e = label_aligned_means(envs, probe)
    if center:
        m_e = m_e - m_e.mean()
    return float((m_e ** 2).sum())


def colored_mnist_probe(side=14, unit_norm=False):
    """Channel-difference probe used throughout Section 4.

    ColoredMNIST images are flattened as [channel0 (side^2), channel1 (side^2)];
    the spurious feature lives in channel0 - channel1. With unit_norm=False
    the probe sums over pixels (the 4.2.1/4.2.2/4.2.4 convention at 14x14);
    with unit_norm=True it averages per pixel, which makes r_hat^2 invariant
    to nearest-neighbour upsampling (the 4.2.5 scalability convention).
    """
    n = side * side
    p = np.concatenate([np.ones(n), -np.ones(n)])
    if unit_norm:
        p = p / n
    return p


# ---------------------------------------------------------------------------
# Algorithm 1 (Section 3.1): blind diagnosis
# ---------------------------------------------------------------------------
def diagnose_blind(envs, min_nr2=100.0):
    """Blind residual diagnosis over the full input space (Algorithm 1).

    No direction is pre-selected. For each environment, the label-aligned
    mean vector v_e = E[x*(2y-1)] is computed on two disjoint halves; the
    residual is the split-half cross estimator (Theorem 1)

        r_hat^2 = sum_e <v_e^(1) - v_bar^(1), v_e^(2) - v_bar^(2)>

    which is unbiased for sum_e ||v_e - v_bar||^2 (the noise term has zero
    mean, so high-dimensional noise does not accumulate a floor; the
    estimator itself can return a small negative value when the true
    residual is below the estimation floor).

    Returns (r2, v_hat, decision, info):
      r2       split-half estimate of the residual
      v_hat    supplementation direction: the smallest-eigenvalue direction
               of the scatter matrix S (full-sample), i.e. the direction of
               near-zero environmental variation (Figure 2 / Lemma A.1);
               when E - 1 < d such directions exist generically
      decision 'yes' iff N_e * r2 >= min_nr2 (the 20%-error floor of
               Theorem 1); 'no' otherwise (supplement along v_hat)
      info     dict with the spurious direction (largest-eigenvalue
               eigenvector), eigenvalues of S, and the half-sample vectors
    """
    halves = []
    for X, y in envs:
        Xn, yn = _np(X), _np(y)
        n = len(yn)
        xa = (Xn[:n // 2] * (2 * yn[:n // 2, None] - 1)).mean(0)
        xb = (Xn[n // 2:] * (2 * yn[n // 2:, None] - 1)).mean(0)
        halves.append((xa, xb))
    V1 = np.stack([h[0] for h in halves])                      # E x d
    V2 = np.stack([h[1] for h in halves])
    r2 = float(((V1 - V1.mean(0)) * (V2 - V2.mean(0))).sum())

    V = (V1 + V2) / 2.0                                        # full-sample means
    W = V - V.mean(0)
    S = W.T @ W                                                # scatter matrix
    evals, evecs = np.linalg.eigh(S)
    v_hat = evecs[:, 0]                                        # smallest: supplementation direction
    v_spu = evecs[:, -1]                                       # largest: spurious direction

    n_e = len(envs[0][1])
    decision = 'yes' if n_e * r2 >= min_nr2 else 'no'
    info = {'v_spurious': v_spu, 'eigenvalues': evals, 'scatter': S,
            'means': V, 'n_per_env': n_e, 'min_nr2': min_nr2}
    return r2, v_hat, decision, info


# ---------------------------------------------------------------------------
# Algorithm 2 (Section 3.2): one-anchor calibration
# ---------------------------------------------------------------------------
def calibrate_c0(lambda_crit_anchor, r2_anchor):
    """Closed form once lambda_crit is known: c0 = lambda_crit * r^2 (scaling law)."""
    return float(lambda_crit_anchor) * float(r2_anchor)


def calibrate_c0_search(candidates, train_eval_fn, probe_grid=(1e2, 1e3, 1e4, 1e5),
                        eps=0.01, min_nr2=100.0):
    """One-anchor calibration of c0 (Algorithm 2).

    candidates    : list of (name, envs) candidate conditions of the problem
                    category; the anchor is the one with the largest blind
                    r_hat^2 satisfying N_e * r_hat^2 >= min_nr2 (Algorithm 1)
    train_eval_fn : callable (envs, lam) -> worst-group accuracy of one
                    training run at penalty lam, all other hyperparameters
                    at their defaults
    probe_grid    : penalty probe grid (paper: {1e2, 1e3, 1e4, 1e5})
    eps           : plateau tolerance (paper: 0.01); lambda_crit is the
                    smallest probed penalty whose accuracy lies within eps
                    of the plateau (the maximum probed accuracy)

    Returns (c0, info) with c0 = lambda_crit * r_hat^2(anchor); info carries
    the anchor name, its r_hat^2, the accuracy curve and lambda_crit.
    """
    best = None
    for name, envs in candidates:
        r2, _, decision, _ = diagnose_blind(envs, min_nr2)
        if decision == 'yes' and (best is None or r2 > best[1]):
            best = (name, r2, envs)
    if best is None:
        raise ValueError('no candidate passes the estimation floor '
                         f'(N_e * r_hat^2 >= {min_nr2:g}); supplement environments')
    name, r2_anchor, envs_anchor = best

    curve = [(lam, float(train_eval_fn(envs_anchor, lam))) for lam in probe_grid]
    plateau = max(a for _, a in curve)
    lam_crit = next(lam for lam, a in curve if a >= plateau - eps)
    c0 = calibrate_c0(lam_crit, r2_anchor)
    info = {'anchor': name, 'r2_anchor': r2_anchor, 'curve': curve,
            'plateau': plateau, 'lambda_crit': lam_crit, 'eps': eps}
    return c0, info


# ---------------------------------------------------------------------------
# Algorithm 3 (Section 3.3): anchored deployment
# ---------------------------------------------------------------------------
def deployment_lambda(c0, r2_hat, safety_factor=2.0):
    """One-off deployment penalty: lambda = safety_factor * c0 / r_hat^2.

    safety_factor = 2 is the safety margin kappa of Section 3.3 (the ablation
    variant `no_safety` corresponds to safety_factor=1).
    """
    return float(safety_factor) * float(c0) / float(r2_hat)


def deploy(envs, c0, probe=None, kappa=2.0, lambda_max=np.inf, min_nr2=100.0,
           **train_kwargs):
    """Anchored deployment (Algorithm 3).

    probe      : None -> blind diagnosis (Algorithm 1); otherwise a known
                 direction (the Section-4 convention on ColoredMNIST)
    kappa      : safety factor (paper: 2)
    lambda_max : trainability ceiling; if the deployed penalty exceeds it,
                 return ('no', ...) instead of training (paper: kappa ~ 5.8
                 relative to the scaling-law value)
    train_kwargs : forwarded to `train_deployment`

    Returns (net, info) on success, (None, info) on refusal; info always
    carries r_hat^2, the decision and, on refusal, the supplementation
    direction v_hat.

    UNIT CONTRACT: c0 must be expressed in the same diagnosis units used here
    -- blind with blind (c0 from `calibrate_c0_search`), probe with probe
    (e.g. the paper's c0 = 51387.4, which is in probe units).
    """
    if probe is None:
        r2, v_hat, decision, diag = diagnose_blind(envs, min_nr2)
    else:
        r2 = diagnose_r2(envs, probe)
        v_hat = np.asarray(probe, dtype=np.float64)
        n_e = len(envs[0][1])
        decision = 'yes' if n_e * r2 >= min_nr2 else 'no'
        diag = {'n_per_env': n_e, 'min_nr2': min_nr2}
    if decision == 'no':
        return None, {'decision': 'no', 'reason': 'below the estimation floor '
                      f'(N_e * r_hat^2 < {min_nr2:g})', 'r2': r2,
                      'v_supplement': v_hat, **diag}
    lam = deployment_lambda(c0, r2, kappa)
    if lam > lambda_max:
        return None, {'decision': 'no', 'reason': f'lambda ({lam:g}) > lambda_max '
                      f'({lambda_max:g})', 'r2': r2, 'lambda': lam,
                      'v_supplement': v_hat, **diag}
    d = envs[0][0].shape[1]
    net = train_deployment(envs, lam, input_dim=d, **train_kwargs)
    return net, {'decision': 'yes', 'r2': r2, 'lambda': lam, **diag}


# ---------------------------------------------------------------------------
# Deployment training (shared by `deploy` and the Section-4 scripts)
# ---------------------------------------------------------------------------
def train_deployment(envs, lam, input_dim, seed=0, steps=400, lr=1e-3,
                     anneal_start=150, anneal_len=100, width=128):
    """One-off deployment training with the annealed IRMv1 penalty.

    Uses the closed-form gradient of the IRMv1 penalty for BCE-with-logits
    (dummy-scalar trick evaluated analytically): for environment risk R_e with
    logits z, grad_penalty = 2 * g_e * d(sigmoid(z))/dz with g_e the squared
    dummy-classifier gradient <sigmoid(z) - y, z> / n.
    """
    torch.manual_seed(seed)
    envs = [(torch.as_tensor(_np(X), dtype=torch.float32),
             torch.as_tensor(_np(y), dtype=torch.float32)) for X, y in envs]
    net = nn.Sequential(nn.Linear(input_dim, width), nn.ReLU(), nn.Linear(width, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for t in range(steps):
        if t < anneal_start:
            lt = 0.0
        elif t < anneal_start + anneal_len:
            lt = lam * (t - anneal_start + 1) / anneal_len
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
def worst_env_accuracy(net, tests):
    """Worst-environment accuracy over a list of test environments."""
    net.eval()
    return min(((net(torch.as_tensor(_np(X), dtype=torch.float32)).squeeze(1) > 0)
                .float() == torch.as_tensor(_np(y), dtype=torch.float32))
               .float().mean().item() for X, y in tests)
