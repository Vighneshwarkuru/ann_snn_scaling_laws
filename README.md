# Inference Complexity Scaling Laws in ANN-to-SNN Conversion

> *"We systematically study how spikes, energy, latency, and accuracy scale in ANN-to-SNN
> conversion under ultra-low-latency inference settings."*

---

## Project Overview

This project derives empirical **scaling laws** for spiking neural network (SNN) inference
complexity by systematically varying:

- **Timesteps T** — `[4, 8, 16, 32, 64]`
- **Network depth** — LeNet, VGG-11, ResNet-18
- **Dataset complexity** — MNIST → Fashion-MNIST → CIFAR-10 → CIFAR-100 → Tiny-ImageNet

It is **not** a new SNN architecture. It is an experimental benchmarking and analysis study.

---

## Pipeline

```
1. Train ANN checkpoints          scripts/train_all_anns.py
2. Run full experiment sweep      scripts/run_sweep.py
3. Aggregate + fit scaling laws   scripts/analyze_results.py
4. Generate all plots             scripts/generate_plots.py
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train all ANN checkpoints (or load pretrained if already saved)
python scripts/train_all_anns.py

# 3. Run the full 225-experiment sweep
python scripts/run_sweep.py

# 4. Analyze results and fit scaling laws
python scripts/analyze_results.py

# 5. Generate all 7 plots
python scripts/generate_plots.py
```

### Dry-run (test the pipeline quickly)
```bash
python scripts/run_sweep.py --dry-run
```
Runs a minimal subset: MNIST / LeNet / T=[4,8] / 1 seed / 3 epochs.

---

## Project Structure

```
ann_snn_scaling_laws/
├── configs/
│   └── experiment.yaml          # Full sweep configuration
├── src/
│   ├── models/                  # LeNet, VGG-11, ResNet-18/34
│   ├── data/                    # Dataset loaders (all 5 datasets)
│   ├── training/                # ANN training + checkpointing
│   ├── conversion/              # ANN→SNN via SpikingJelly
│   ├── evaluation/              # Accuracy, spike logging, energy
│   ├── analysis/                # Scaling law fitting, SCI/CSI
│   └── visualization/           # All 7 plots
├── scripts/
│   ├── train_all_anns.py
│   ├── run_sweep.py
│   ├── analyze_results.py
│   └── generate_plots.py
├── results/
│   ├── raw/                     # Per-run CSVs
│   ├── aggregated/              # Mean ± std across seeds
│   ├── plots/                   # PNG + PDF plots
│   └── scaling_laws/            # Fitted equations + R² values
├── checkpoints/                 # Saved ANN weights
└── data/                        # Downloaded datasets
```

---

## Metrics Collected

| Metric | Description |
|--------|-------------|
| Accuracy | ANN and SNN top-1 classification accuracy |
| Spike Count | Total and per-layer spikes per inference sample |
| Spike Density | `Total Spikes / Total Neurons` |
| Latency | Wall-clock inference time (ms/sample) |
| Energy | `E_SNN ≈ Spikes × 0.9 pJ`, `E_ANN ≈ MACs × 4.6 pJ` |
| SynOps | Accumulate operations triggered by spikes |
| Memory | Peak memory usage during inference |

---

## Novel Indices

**Spike Complexity Index (SCI)**
```
SCI = Total_Spikes / Accuracy
```
Lower SCI = more efficient use of spikes per unit accuracy.

**Complexity Scaling Index (CSI)**
```
CSI = (ΔEnergy + ΔLatency) / ΔAccuracy
```
Measures the marginal complexity cost of accuracy gain when doubling T.

---

## Scaling Laws Fitted

| Relationship | Model | Formula |
|---|---|---|
| Accuracy vs T | Logarithmic | `Acc(T) = a·log₂(T) + b` |
| Spikes vs T | Linear | `Spikes(T) = c·T + d` |
| Energy vs Depth | Linear / Power-law | Best fit selected by R² |
| Spike Density vs Dataset | Linear / Power-law | Best fit selected by R² |

---

## Configuration

Edit `configs/experiment.yaml` to change the sweep:

```yaml
sweep:
  datasets: [mnist, fashion_mnist, cifar10, cifar100, tiny_imagenet]
  models: [lenet, vgg11, resnet18]
  timesteps: [4, 8, 16, 32, 64]
  seeds: [42, 123, 7]

training:
  epochs: 50
  batch_size: 128
  lr: 0.01

evaluation:
  device: auto   # auto, cuda, mps, cpu
```

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- SpikingJelly 0.0.0.0.14+
- See `requirements.txt` for full list

---

## Outputs

After a full run you'll find:

- `results/plots/` — 7 publication-ready plots (PNG + PDF)
- `results/aggregated/aggregated_results.csv` — mean ± std metrics
- `results/scaling_laws/scaling_law_fits.json` — fitted equations and R² values
- `checkpoints/` — trained ANN weights for reuse
