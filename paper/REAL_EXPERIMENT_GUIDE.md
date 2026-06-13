# Running Real Experiments: Guide for Your MacBook

## Status: Pipeline Verified ✅

The full pipeline works end-to-end with **real training, real conversion, real evaluation**:

```
✅ src/data/datasets.py — fully implemented (MNIST, Fashion-MNIST, CIFAR-10/100, Tiny-ImageNet)
✅ src/models/ — VGG-9/11/16, ResNet-18/34, LeNet all implemented
✅ src/training/train_ann.py — full training loop with checkpointing
✅ src/conversion/ann_to_snn.py — SpikingJelly ann2snn conversion
✅ src/evaluation/ — SpikeLogger, energy model, MAC counting (ptflops)
✅ scripts/run_sweep.py — tested, produces real CSV results
✅ scripts/analyze_results.py — scaling law fitting
✅ scripts/generate_plots.py — visualization pipeline
```

## Your Machine Specs

- **Device:** Apple MacBook (MPS available)
- **PyTorch:** 2.8.0
- **SpikingJelly:** ✅ installed
- **ptflops:** ✅ installed

## Runtime Estimates

| Configuration | Experiments | Est. Time (MPS) | Est. Time (CPU) |
|---------------|-------------|------------------|-----------------|
| `--fast` (MNIST/VGG-9/T=[4,16,64]) | 3 | ~10 min | ~20 min |
| Standard (3 datasets/2 models/6 T) | 36 | ~2-3 hrs | ~5-6 hrs |
| Full (5 datasets/5 models/8 T/3 seeds) | 600 | ~24-48 hrs | Not recommended |

## How to Run

### Option 1: Fast Verification (~10 min)
```bash
python scripts/run_real_experiment.py --fast
```
Trains VGG-9 on MNIST (10 epochs), converts to SNN, tests T=[4,16,64].
Proves the scaling pattern (accuracy increases with T) with real data.

### Option 2: Standard Experiment (~2-3 hrs)
```bash
python scripts/run_real_experiment.py
```
Tests 2 models × 3 datasets × 6 timesteps = 36 experiments.
Enough data to fit scaling laws and verify CSI/SCI.

### Option 3: Full Sweep (overnight, ~24 hrs)
```bash
python scripts/run_sweep.py --config configs/experiment_cpu.yaml
```
Full factorial design with multiple seeds.

### Option 4: Use Google Colab (free GPU, ~2-4 hrs for full sweep)
Upload your code to Colab with a T4 GPU:
```python
!git clone https://github.com/Vighneshwarkuru/ann_snn_scaling_laws.git
%cd ann_snn_scaling_laws
!pip install -r requirements.txt
!python scripts/run_sweep.py --config configs/experiment.yaml
```

## After Running: Analyze Results

```bash
# Analyze and fit scaling laws
python scripts/analyze_results.py --results-dir results_real

# Generate plots
python scripts/generate_plots.py --results-dir results_real
```

## What to Expect from Real Data

### Scaling Laws Should Hold:
1. **Accuracy vs T:** Logarithmic saturation (Acc = a·log₂(T) + b)
2. **Spikes vs T:** Linear growth (Spikes ≈ k·T)
3. **Energy vs T:** Linear growth (E_SNN = 0.9 × Spikes)
4. **CSI power law:** CSI(T,D,S) = α·T^β·D^γ·S^δ

### Differences from Synthetic Data:
- ANN accuracies will differ slightly (depends on training)
- Conversion loss will be real (not idealized logarithmic)
- Some (model, dataset) combos may fail conversion (torch.fx issues)
- Spike counts reflect actual neuron firing patterns
- Energy crossover point will be at different T

### Key Scientific Questions:
1. Does accuracy really follow log₂(T)? → Fit R² will tell you
2. Is CSI really a power law? → Compare R² of power vs linear vs exponential
3. Which exponent dominates (β vs γ vs δ)? → Sensitivity analysis
4. Where does energy crossover happen? → Plot E_SNN vs E_ANN

## Known Issues

1. **MPS + SpikingJelly:** Conversion must happen on CPU (torch.fx limitation).
   SNN evaluation also runs on CPU for stability.

2. **LeNet checkpoint:** The existing `lenet_mnist.pth` was trained for only 3 epochs
   (86.63% accuracy). Delete it and retrain for better results:
   ```bash
   rm checkpoints/lenet_mnist.pth
   ```

3. **Tiny-ImageNet:** Downloads ~240MB. First run will be slow.

4. **Memory:** ResNet-34 on Tiny-ImageNet may use 4-6 GB RAM during conversion.
