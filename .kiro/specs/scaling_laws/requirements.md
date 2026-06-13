# Requirements: Inference Complexity Scaling Laws in ANN-to-SNN Conversion

## Overview

This project systematically studies how inference complexity — measured as spikes, energy,
latency, and accuracy — scales in ANN-to-SNN conversion across varying timesteps, network
depths, and dataset complexities. The goal is to derive empirical scaling laws and define
new efficiency indices, not to invent a new SNN architecture.

---

## Requirements

### REQ-1: ANN Model Support

The system SHALL support training and/or loading pretrained ANN models across multiple
architecture families:

- **REQ-1.1** LeNet (shallow CNN, easy baseline)
- **REQ-1.2** VGG-11 (medium depth)
- **REQ-1.3** ResNet-18 (medium depth, skip connections)
- **REQ-1.4** ResNet-34 (deeper variant)
- **REQ-1.5** MobileNet (lightweight, advanced)

Each model SHALL be trainable from scratch and loadable from a pretrained checkpoint.
ANN accuracy SHALL be recorded before conversion as a baseline.

---

### REQ-2: ANN-to-SNN Conversion

The system SHALL convert trained ANN models to SNNs using at least one of the following
frameworks:

- **REQ-2.1** SpikingJelly (primary, preferred for rate-coding conversion)
- **REQ-2.2** SNNTorch (secondary option)

Conversion SHALL use rate-coding with threshold balancing / layer-wise normalization.
The converted SNN SHALL be functionally equivalent to the source ANN (same architecture,
same weights rescaled for spiking dynamics).

---

### REQ-3: Dataset Support

The system SHALL support the following datasets spanning a complexity gradient:

| Dataset       | Classes | Complexity |
|---------------|---------|------------|
| MNIST         | 10      | Very Easy  |
| Fashion-MNIST | 10      | Easy       |
| CIFAR-10      | 10      | Medium     |
| CIFAR-100     | 100     | Hard       |
| Tiny-ImageNet | 200     | Very Hard  |

- **REQ-3.1** Each dataset SHALL be downloadable automatically via standard PyTorch loaders.
- **REQ-3.2** Preprocessing (normalization, resizing) SHALL be standardized per dataset.

---

### REQ-4: Experiment Parameter Grid

The system SHALL support a fully configurable experiment matrix sweeping the following
independent variables:

- **REQ-4.1 Timesteps (T):** `[4, 8, 16, 32, 64]`
- **REQ-4.2 Models:** at minimum `[LeNet, VGG-11, ResNet-18]`
- **REQ-4.3 Datasets:** all 5 listed in REQ-3
- **REQ-4.4 Random Seeds:** 3 seeds per configuration for statistical reliability

Minimum experiment count: 5 datasets × 3 models × 5 timesteps × 3 seeds = **225 runs**.

All experiment configurations SHALL be defined in a single YAML config file and executed
programmatically without manual intervention per run.

---

### REQ-5: Metric Collection

The system SHALL measure and log the following metrics for every experiment run:

#### REQ-5.1 — Accuracy
- ANN top-1 accuracy (pre-conversion baseline)
- SNN top-1 accuracy at each timestep T
- Accuracy drop: `ΔAcc = ANN_acc − SNN_acc`

#### REQ-5.2 — Spike Count
- Total spikes across all layers per inference sample
- Average spikes per sample over the test set
- Per-layer spike counts
- Spikes per neuron per timestep

#### REQ-5.3 — Spike Density
- `Spike Density = Total Spikes / Total Neurons`
- Computed globally and per layer

#### REQ-5.4 — Inference Latency
- Wall-clock inference time per sample (ms)
- Total inference time over the test set
- Measured separately for ANN and SNN

#### REQ-5.5 — Synaptic Operations (SynOps)
- ANN: Multiply-Accumulate operations (MACs)
- SNN: Accumulate operations (ACs) triggered by spikes
- Ratio: `SynOps_SNN / SynOps_ANN`

#### REQ-5.6 — Energy Estimation
- Estimated energy using the approximation:
  `E_SNN ≈ Spike_Count × E_AC` where `E_AC = 0.9 pJ`
  `E_ANN ≈ MAC_Count × E_MAC` where `E_MAC = 4.6 pJ`
- (Constants from Horowitz 2014, standard in neuromorphic literature)
- Energy ratio: `E_SNN / E_ANN`

#### REQ-5.7 — Memory Usage
- Peak GPU/CPU memory during inference (MB)
- Spike buffer memory

All metrics SHALL be written to a structured CSV/JSON results file per run.

---

### REQ-6: Scaling Indices

The system SHALL compute the following derived indices from collected metrics:

#### REQ-6.1 — Spike Complexity Index (SCI)
```
SCI = Total_Spikes / Accuracy
```
Lower SCI indicates better spike efficiency. Computed per (model, dataset, T) combination.

#### REQ-6.2 — Complexity Scaling Index (CSI)
```
CSI = (ΔEnergy + ΔLatency) / ΔAccuracy
```
Measures how much complexity grows per unit of accuracy gain when increasing T.
Computed as a finite difference between consecutive timestep settings.

---

### REQ-7: Scaling Law Analysis

The system SHALL fit and report empirical scaling relationships:

- **REQ-7.1** Accuracy vs. T: fit a logarithmic saturation curve `Acc(T) = a·log(T) + b`
- **REQ-7.2** Spike Count vs. T: fit a linear model `Spikes(T) = c·T + d`
- **REQ-7.3** Energy vs. Depth: fit linear and power-law models, report best fit
- **REQ-7.4** Spike Density vs. Dataset complexity: ranked comparison across the 5 datasets

Fitted parameters and R² values SHALL be reported for each scaling relationship.

---

### REQ-8: Visualization

The system SHALL produce the following plots (saved as PDF and PNG):

| Plot ID | X-Axis         | Y-Axis          | Grouping        |
|---------|----------------|-----------------|-----------------|
| P1      | Timesteps T    | Accuracy (%)    | Model & Dataset |
| P2      | Timesteps T    | Energy (pJ)     | Model & Dataset |
| P3      | Spike Count    | Accuracy (%)    | Model & Dataset |
| P4      | Network Depth  | Latency (ms)    | Dataset         |
| P5      | Dataset        | Energy (pJ)     | Model & T       |
| P6      | Timesteps T    | SCI             | Model           |
| P7      | Timesteps T    | CSI             | Model           |

All plots SHALL include error bars (std dev across seeds) and fitted scaling curves where applicable.

---

### REQ-9: Experiment Reproducibility

- **REQ-9.1** All random seeds SHALL be fixed and logged per run.
- **REQ-9.2** All hyperparameters (batch size, epochs, LR, conversion threshold) SHALL be
  logged in the results file.
- **REQ-9.3** Model checkpoints SHALL be saved after ANN training and reused across SNN
  conversion runs to eliminate training variance.

---

### REQ-10: Output Artifacts

Each completed experiment sweep SHALL produce:

- `results/raw/` — per-run CSVs with all metrics
- `results/aggregated/` — mean ± std across seeds, per (model, dataset, T)
- `results/plots/` — all 7 plots listed in REQ-8
- `results/scaling_laws/` — fitted curve parameters and R² values
- `checkpoints/` — trained ANN weights per (model, dataset)

---

## Success Criteria

| Criterion | Target |
|-----------|--------|
| SNN accuracy drop vs. ANN | < 5% at T=32 for CIFAR-10 |
| Spike density | Measurable and varying across datasets |
| Scaling law fit quality | R² > 0.85 for T vs. Accuracy and T vs. Spikes |
| CSI and SCI | Computed for all 225 runs |
| All 7 plots generated | Without manual post-processing |
| Full 225-run sweep | Completable end-to-end via single entry point |
