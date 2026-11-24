# BIRD: Blind IRM Residual Diagnostic — Experiment Code

Official experiment code accompanying the paper. BIRD is a one-off alternative
to hyperparameter search for IRM-style penalized training: it (i) measures the
cross-environment residual magnitude r̂² directly from the training
environments (no model training, milliseconds), (ii) converts it to a
deployment penalty through the scaling law λ = κ·c₀/r̂² with a constant c₀
calibrated at a **single** anchor condition and safety factor κ = 2, and
(iii) trains **once**. The experiments below compare this route against
grid-searched DomainBed baselines (ERM / IRM / VREx / IB-IRM) on a controlled
ColoredMNIST benchmark (Section 4 of the paper).

## Repository layout

```
bird.py                                 Reference implementation of BIRD's three
                                        algorithms (Section 3): Algorithm 1 blind
                                        diagnosis, Algorithm 2 one-anchor
                                        calibration, Algorithm 3 anchored
                                        deployment, for use on your own
                                        multi-environment data.
experiments/
  fig5_residual_scan.py                 Section 4.2.1 / Figure 5: residual scan
                                        (Delta in {0.05, 0.10, 0.20, 0.30},
                                        r_hat^2 ~ 2.4 .. 119.5)
  fig6_center_scan.py                   Section 4.2.2 / Figure 6: correlation-centre
                                        scan (centre in {0.75, 0.80, 0.85}, Delta = 0.20)
  fig7_ablation.py                      Section 4.2.4 / Figure 7: ablation
                                        (full / no_align / no_center / no_scalelaw / no_safety)
  fig8_scalability.py                   Section 4.2.5 / Figure 8: scalability study
                                        (input side 14 -> 140, d = 392 .. 39200)
  mechanism_gradient_binding.py         Companion to 4.2.5: bitwise evidence for the
                                        Adam x nearest-neighbour gradient-binding mechanism
  mechanism_layered_lr.py               Companion to 4.2.5: layered-learning-rate
                                        equivalence verdict (the fix used by fig8)
  certify_monotonicity.py               Appendix A.4, Corollary A.6: rigorous
                                        interval-arithmetic certificate that
                                        c(x) = lambda*(x)*x is strictly decreasing
                                        on the scalar slice (runs in seconds)
plotting/
  plot_fig5.py  plot_fig6.py  plot_fig7.py  plot_fig8.py
                                        Regenerate the paper figures from the CSVs.
results/
  fig5_residual_scan.csv                Paper-reported aggregated values (Figure 5)
  fig6_center_scan.csv                  Paper-reported aggregated values (Figure 6)
  fig7_ablation.csv                     Paper-reported raw per-run rows (Figure 7)
  fig8_scalability.csv                  Paper-reported raw per-run rows (Figure 8)
figures/                                Output of the plotting scripts.
requirements.txt
```

The experiment scripts are **self-contained** (each carries its own copy of the
data-construction and training code, exactly as run for the paper). This is
deliberate: what you run is what produced the reported numbers.

## Installation

Python >= 3.10 (paper reports Python 3.11 / PyTorch 2.13). All experiments are
CPU-only by design (the paper's measurement protocol).

```bash
pip install -r requirements.txt
```

The baseline experiments (fig5, fig6, and the ERM rows of fig8) use the
**official DomainBed classes**, which are not on PyPI:

```bash
git clone --depth 1 https://github.com/facebookresearch/DomainBed.git
```

Clone it into the repository root, or point the environment variable
`DOMAINBED_PATH` at an existing checkout. The BIRD-only paths (diagnosis,
deployments, ablation, mechanism probes, certificate) do **not** need
DomainBed; `--no-baselines` skips the baseline grid entirely.

MNIST is downloaded automatically by torchvision on first run (into `data/`).
If the automatic mirrors are unreachable, place the four raw ubyte files under
`data/MNIST/raw/` manually.

## Quick start

```bash
# Appendix certificate — a few seconds, needs no data
python experiments/certify_monotonicity.py

# Ablation anchor check — ~1 min, downloads MNIST on first run
python experiments/fig7_ablation.py --smoke
# expect: diag r^2 = 51.3874 (lambda = 2000), deploy worst-env acc ~ 0.633-0.634

# Scalability anchor check (BIRD path) — ~1 min
python experiments/fig8_scalability.py --smoke

# Regenerate all figures from the paper-reported CSVs — seconds
python plotting/plot_fig5.py
python plotting/plot_fig6.py
python plotting/plot_fig7.py
python plotting/plot_fig8.py
```

## Reproducing the paper experiments

All scripts accept `--data-dir` (MNIST location, default `./data`) and
`--output-dir` (default `./results_generated`, so the paper-reported CSVs in
`results/` are never overwritten). All long experiments run each measurement
in a separate subprocess (clean per-run peak memory) and support **checkpoint
resume**: re-running a script skips completed rows.

| Script | Paper | Jobs | Reference runtime (CPU) |
|---|---|---|---|
| `experiments/fig5_residual_scan.py` | §4.2.1, Figure 5 | 4 gaps x (1 diag + 2 BIRD + 2 ERM + 36 grid) | ~1.5-2 h |
| `experiments/fig6_center_scan.py` | §4.2.2, Figure 6 | 3 centres x (1 diag + 2 BIRD + 2 ERM + 36 grid) | ~1-1.5 h |
| `experiments/fig7_ablation.py` | §4.2.4, Figure 7 | 5 variants x 3 gaps x (1 diag + 2 deploy) | ~20-30 min |
| `experiments/fig8_scalability.py` | §4.2.5, Figure 8 | 2 scales x (diag + 2 BIRD + 2 ERM) | ~2-2.5 h |
| `experiments/mechanism_gradient_binding.py` | §4.2.5 companion | tie check + 8 training runs | ~15 min |
| `experiments/mechanism_layered_lr.py` | §4.2.5 companion | 4 training runs | ~10 min |
| `experiments/certify_monotonicity.py` | Appendix A.4, Cor. A.6 | 155 interval segments | seconds |

After a full run, the aggregated table (fig5/fig6) or raw rows (fig7/fig8) land
in `results_generated/`; point the plot scripts at them with `--csv`, e.g.

```bash
python plotting/plot_fig7.py --csv results_generated/fig7_ablation_raw.csv
```

## Using BIRD on your own data

`bird.py` implements the three algorithms of Section 3 one-to-one:

```python
import bird

envs = [(X1, y1), (X2, y2)]                     # X: (n, d) float tensor, y in {0, 1}

# Algorithm 1 — blind diagnosis (no pre-selected direction)
r2, v_hat, decision, info = bird.diagnose_blind(envs)
#   r2: split-half unbiased residual estimate; v_hat: supplementation
#   direction; decision: 'yes' iff N_e * r2 >= 100 (Theorem 1 floor)

# Algorithm 2 — one-anchor calibration of the class constant c0
c0, cal = bird.calibrate_c0_search(
    candidates=[('cond-A', envs)],              # anchor = largest r2 above floor
    train_eval_fn=my_train_fn,                  # (envs, lam) -> worst-group acc
    probe_grid=(1e2, 1e3, 1e4, 1e5), eps=0.01)  # paper's grid and tolerance

# Algorithm 3 — anchored deployment (trains ONCE, or refuses with v_hat)
net, out = bird.deploy(envs, c0, kappa=2.0, lambda_max=1e6)
if net is None:
    ...                                         # out['reason'], out['v_supplement']
acc = bird.worst_env_accuracy(net, test_envs)
```

For ColoredMNIST-style problems where the spurious direction is known, use
the probe path of Section 4 instead: `bird.colored_mnist_probe()`,
`bird.diagnose_r2`, `bird.deployment_lambda`, `bird.train_deployment`.
**Unit contract**: c0 always carries the units of the diagnosis it was
calibrated with — the paper's c0 = 51387.4 is in *probe* units and only
combines with `diagnose_r2` / `deploy(probe=...)`; a c0 from
`calibrate_c0_search` is in *blind* units and only combines with
`deploy(probe=None)`.

## Paper-reported results (headline numbers)

The CSVs in `results/` are the exact values reported in the paper. Fresh runs
reproduce the accuracy columns up to small cross-version numeric variation
(see notes below); time and memory are hardware-dependent.

**Figure 5 (§4.2.1, residual scan)** — worst-env accuracy, BIRD one-off vs best
grid-searched baseline; BIRD time = diagnosis + 2 deployments, baselines =
whole grid:

| r̂² | BIRD | ERM | IRM | VREx | IB-IRM | BIRD time | IRM grid time |
|---|---|---|---|---|---|---|---|
| 2.37 | 0.373 | 0.294 | 0.435 | 0.310 | 0.506 | 22.3 s | 223.7 s |
| 12.09 | 0.598 | 0.280 | 0.426 | 0.497 | 0.502 | 22.2 s | 223.6 s |
| 51.39 | 0.644 | 0.297 | 0.435 | 0.631 | 0.517 | 21.2 s | 235.9 s |
| 119.49 | 0.575 | 0.303 | 0.460 | 0.615 | 0.504 | 21.5 s | 219.8 s |

**Figure 6 (§4.2.2, centre scan, Δ = 0.20)** — BIRD 0.542 / 0.582 / 0.643 at
centres 0.75 / 0.80 / 0.85 (r̂² stays ~ 41-52 throughout).

**Figure 7 (§4.2.4, ablation)** — worst-env accuracy, seed mean:

| variant | Δ = 0.10 | Δ = 0.20 | Δ = 0.30 |
|---|---|---|---|
| full | 0.5992 | 0.6432 | 0.5720 |
| w/o label alignment | 0.5674 | 0.5204 | 0.4928 |
| w/o centering | 0.2484 | 0.4182 | 0.5502 |
| w/o scaling law | 0.3322 | 0.6424 | 0.5038 |
| w/o safety factor | 0.4614 | 0.6078 | 0.6276 |

**Figure 8 (§4.2.5, scalability)** — measured at d = 392 (14x14) and
d = 39200 (140x140); BIRD per-seed worst-env accuracy 0.6348 / 0.6540 (14x14)
and 0.6360 / 0.6624 (140x140), with λ = 2000 and r̂² ≈ 0.001338 identical at
both scales. Official ERM: 0.3012 / 0.2944 (14x14), 0.2732 / 0.2824 (140x140).

## Measurement protocol and reproducibility notes

- **Metric conventions** (§4.1.4): accuracy = worst-environment test accuracy
  (test correlations 0.1 / 0.9); time = wall time of training + tuning budget
  (for BIRD: diagnosis + deployments; for baselines: the full λ grid); memory
  = per-run peak RSS (fig5/6/7) and absolute process peak (fig8), measured in
  isolated subprocesses.
- **Determinism**: data splits (rng 123), environment draws
  (1000·seed + 100·Δ) and training (torch seed = deployment seed) are all
  fixed. Accuracy is deterministic for a fixed software stack; across
  PyTorch/BLAS versions we observe worst-env accuracy differences of up to
  ~0.01 (e.g. the anchor deployment reports 0.6328 in the paper's environment
  and 0.6344 under PyTorch 2.8). Time and memory scale with hardware.
- **CPU-only**: the scripts never touch the GPU; this matches the paper's
  measurement protocol and keeps results comparable across machines.
- **Windows**: fully supported (no POSIX-only APIs); peak memory is read via
  `psutil` (peak working set) where the POSIX `resource` module is unavailable.
- **Checkpoint resume**: partial results are appended to the raw CSV after
  every job; re-running a script skips completed rows, so interrupted runs
  continue where they stopped.
- **Projected point in Figure 8**: d = 3.92×10⁶ (side 1400) is *not* measured
  (≈ 235 GB of float32 data, ≈ 33 h per CPU run). The time/memory panels show
  a linear-in-d projection through the two measured points (dotted line,
  hollow marker, grey "projected" band); the accuracy panel flat-extends the
  measured value, justified by the function-space equivalence established by
  the two mechanism scripts — not by trend fitting. See the docstring of
  `plotting/plot_fig8.py` and the paper caption.
- **Anchor self-checks**: `fig7_ablation.py --smoke` must print
  `r^2 = 51.3874`, `lambda = 2000`, acc ≈ 0.63; `fig8_scalability.py --smoke`
  must print `r^2 ≈ 0.0013`, `lambda = 2000`. If these anchors hold, the whole
  pipeline is byte-faithful to the paper runs.

## Citation

If you use this code, please cite the paper (citation entry to be added upon
publication).
