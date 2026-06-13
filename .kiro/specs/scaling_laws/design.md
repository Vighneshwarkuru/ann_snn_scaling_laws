# Design: Inference Complexity Scaling Laws in ANN-to-SNN Conversion

## Technology Stack

| Component         | Choice                              | Reason                                      |
|-------------------|-------------------------------------|---------------------------------------------|
| Deep Learning     | PyTorch 2.x                         | Standard, SpikingJelly native support       |
| SNN Conversion    | SpikingJelly 0.0.0.0.14+            | Best-in-class rate-coding conversion tools  |
| Data              | torchvision + custom Tiny-ImageNet  | All datasets except Tiny-ImageNet built-in  |
| Config Management | Hydra / plain YAML                  | Reproducible sweep configs                  |
| Experiment Runner | Python multiprocessing / simple loop| Avoid heavy MLflow dependency               |
| Results Storage   | CSV + JSON                          | Lightweight, no DB required                 |
| Visualization     | Matplotlib + Seaborn                | Standard scientific plotting                |
| Curve Fitting     | SciPy                               | Logarithmic and power-law fits              |

---

## Project Structure

```
ann_snn_scaling_laws/
├── configs/
│   ├── experiment.yaml          # Master sweep config
│   ├── models/
│   │   ├── lenet.yaml
│   │   ├── vgg11.yaml
│   │   └── resnet18.yaml
│   └── datasets/
│       ├── mnist.yaml
│       ├── fashion_mnist.yaml
│       ├── cifar10.yaml
│       ├── cifar100.yaml
│       └── tiny_imagenet.yaml
│
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lenet.py             # LeNet ANN definition
│   │   ├── vgg.py               # VGG-11 ANN definition
│   │   └── resnet.py            # ResNet-18/34 ANN definition
│   │
│   ├── conversion/
│   │   ├── __init__.py
│   │   └── ann_to_snn.py        # SpikingJelly conversion wrapper
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── datasets.py          # Dataset loaders and transforms
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── train_ann.py         # ANN training loop
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate_ann.py      # ANN accuracy evaluation
│   │   ├── evaluate_snn.py      # SNN accuracy + spike logging
│   │   └── metrics.py           # All metric computation
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── scaling_laws.py      # Curve fitting and law derivation
│   │   └── indices.py           # CSI and SCI computation
│   │
│   └── visualization/
│       ├── __init__.py
│       └── plots.py             # All 7 required plots
│
├── scripts/
│   ├── train_all_anns.py        # Train/save ANN checkpoints for all (model, dataset)
│   ├── run_sweep.py             # Main experiment sweep runner
│   ├── analyze_results.py       # Post-hoc scaling law fitting
│   └── generate_plots.py        # Regenerate all plots from results
│
├── results/
│   ├── raw/                     # Per-run CSVs
│   ├── aggregated/              # Mean ± std across seeds
│   ├── plots/                   # PDF + PNG outputs
│   └── scaling_laws/            # Fitted parameters and R² values
│
├── checkpoints/                 # Saved ANN weights
├── data/                        # Downloaded datasets
├── requirements.txt
└── README.md
```

---

## Module Design

### 1. `src/models/`

Each model file defines a standard `nn.Module` subclass.

**LeNet** (`lenet.py`)
- Two conv layers + two FC layers
- Batch norm after each conv (required for threshold normalization)
- Configurable for MNIST (1-channel) and CIFAR (3-channel)

**VGG-11** (`vgg.py`)
- Standard VGG-11 block structure
- Replace all MaxPool with AvgPool (required for SpikingJelly conversion)
- Batch norm after every conv

**ResNet-18** (`resnet.py`)
- Standard ResNet-18 with BasicBlock
- Skip connections handled via SpikingJelly's identity shortcut support
- Batch norm after every conv

> **Design Note:** All models use ReLU activations and BatchNorm. These are the two
> prerequisites for clean threshold-based ANN→SNN conversion in SpikingJelly.

---

### 2. `src/conversion/ann_to_snn.py`

Wraps SpikingJelly's `ann2snn` pipeline:

```python
class ANNtoSNNConverter:
    def __init__(self, model: nn.Module, T: int, norm_percentile: float = 0.9995):
        ...
    
    def convert(self, calibration_loader: DataLoader) -> nn.Module:
        # 1. Run calibration forward passes to collect activation statistics
        # 2. Compute per-layer thresholds via percentile normalization
        # 3. Replace ReLU layers with LIF neurons (SpikingJelly IFNode/LIFNode)
        # 4. Set threshold per neuron layer
        # Returns: converted SNN module
        ...
    
    def set_timesteps(self, T: int) -> None:
        # Update T on all SNN temporal layers
        ...
```

The SNN operates in **rate coding** mode: input images are repeated T times as constant
current input. Spikes are accumulated over T steps to produce the output logit.

---

### 3. `src/data/datasets.py`

```python
def get_dataloaders(dataset_name: str, batch_size: int, seed: int) 
    -> Tuple[DataLoader, DataLoader]:
    # Returns (train_loader, test_loader)
    # Handles: MNIST, FashionMNIST, CIFAR10, CIFAR100, TinyImageNet
```

Tiny-ImageNet requires a custom loader since it's not in torchvision. The loader downloads
from [http://cs231n.stanford.edu/tiny-imagenet-200.zip](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
and organizes into ImageFolder format.

All datasets are normalized to zero mean / unit variance using dataset-specific statistics.

---

### 4. `src/evaluation/metrics.py`

Central metric computation. The `SpikeLogger` uses PyTorch forward hooks:

```python
class SpikeLogger:
    def __init__(self, snn_model: nn.Module):
        self.hooks = []
        self.layer_spikes: Dict[str, List[int]] = {}
        self._register_hooks(snn_model)
    
    def _register_hooks(self, model):
        # Attach forward hooks to all SpikingJelly spiking layers
        # Hooks count non-zero outputs (spikes) per forward pass
        ...
    
    def reset(self):
        self.layer_spikes = {}
    
    def get_total_spikes(self) -> int: ...
    def get_per_layer_spikes(self) -> Dict[str, int]: ...
    def get_spike_density(self, total_neurons: int) -> float: ...
```

Energy estimation:

```python
E_AC  = 0.9e-12   # Joules, accumulate op on 45nm (Horowitz 2014)
E_MAC = 4.6e-12   # Joules, multiply-accumulate op on 45nm

def estimate_energy(spike_count: int, synops_snn: int, mac_count_ann: int) -> Dict:
    e_snn = spike_count * E_AC
    e_ann = mac_count_ann * E_MAC
    return {"E_SNN": e_snn, "E_ANN": e_ann, "ratio": e_snn / e_ann}
```

MAC count for ANN layers is computed once via `ptflops` or manual convolution arithmetic.

---

### 5. `src/analysis/scaling_laws.py`

Fits empirical scaling relationships using `scipy.optimize.curve_fit`:

```python
def fit_accuracy_vs_T(T_values, acc_values) -> FitResult:
    # Fits: Acc(T) = a * log2(T) + b
    # Returns: params (a, b), R², fitted curve
    ...

def fit_spikes_vs_T(T_values, spike_values) -> FitResult:
    # Fits: Spikes(T) = c * T + d  (linear)
    # Returns: params (c, d), R²
    ...

def fit_energy_vs_depth(depth_values, energy_values) -> FitResult:
    # Fits both linear and power law, returns best R²
    ...
```

---

### 6. `src/analysis/indices.py`

```python
def compute_SCI(total_spikes: float, accuracy: float) -> float:
    return total_spikes / accuracy

def compute_CSI(delta_energy: float, delta_latency: float, delta_accuracy: float) -> float:
    if delta_accuracy == 0:
        return float('inf')
    return (delta_energy + delta_latency) / delta_accuracy
```

CSI is computed as a finite difference between consecutive T values:
`CSI(T→2T) = (E(2T) - E(T) + L(2T) - L(T)) / (Acc(2T) - Acc(T))`

---

### 7. `scripts/run_sweep.py`

The main entry point. Reads `configs/experiment.yaml` and iterates over all combinations:

```
for dataset in datasets:
    for model in models:
        load ANN checkpoint (train if missing)
        for seed in seeds:
            for T in timesteps:
                convert ANN → SNN with timestep T
                evaluate SNN on test set
                log: accuracy, spike_count, latency, energy, SCI, CSI
                save to results/raw/{model}_{dataset}_T{T}_seed{seed}.csv
```

---

## Data Flow

```
[configs/experiment.yaml]
         │
         ▼
[scripts/run_sweep.py]
         │
    ┌────┴────┐
    │         │
    ▼         ▼
[train_ann] [load checkpoint]
    │
    ▼
[ANNtoSNNConverter.convert(T)]
    │
    ▼
[evaluate_snn + SpikeLogger]
    │
    ├─→ accuracy
    ├─→ spike counts (total, per-layer)
    ├─→ spike density
    ├─→ inference latency
    ├─→ SynOps (AC)
    ├─→ energy estimate
    └─→ memory usage
         │
         ▼
[results/raw/*.csv]
         │
         ▼
[analyze_results.py]
    ├─→ scaling law fits
    ├─→ CSI / SCI per run
    └─→ results/aggregated/*.csv
         │
         ▼
[generate_plots.py]
    └─→ results/plots/*.pdf + *.png
```

---

## Configuration Schema (`configs/experiment.yaml`)

```yaml
experiment:
  name: "ann_snn_scaling_laws_v1"
  output_dir: "results"
  checkpoint_dir: "checkpoints"

sweep:
  datasets: [mnist, fashion_mnist, cifar10, cifar100, tiny_imagenet]
  models: [lenet, vgg11, resnet18]
  timesteps: [4, 8, 16, 32, 64]
  seeds: [42, 123, 7]

training:
  epochs: 50
  batch_size: 128
  lr: 0.01
  weight_decay: 5e-4
  scheduler: cosine

conversion:
  norm_percentile: 0.9995
  calibration_batches: 8

evaluation:
  batch_size: 256
  device: cuda  # or cpu
```

---

## Key Design Decisions

**Why SpikingJelly over SNNTorch?**
SpikingJelly's `ann2snn` module provides automated threshold normalization from pretrained
ANNs, which is exactly what this project needs. SNNTorch is better suited for
training-from-scratch with surrogate gradients.

**Why rate coding?**
Rate coding (repeat input T times, accumulate spikes) is the standard for ANN-to-SNN
conversion. It makes the timestep T a clean independent variable directly tied to
accuracy and spike count.

**Why AvgPool instead of MaxPool in VGG?**
SpikingJelly's threshold normalization assumes linear operations. MaxPool introduces a
non-linearity that breaks the normalization assumptions. AvgPool is a standard fix in
ANN-to-SNN literature.

**Why save ANN checkpoints before the sweep?**
The 225-run sweep should only vary T and seeds, not ANN weights. Training variance would
confound the scaling law analysis. One ANN per (model, dataset) is trained once, saved,
and reused across all T and seed variations.

**Why finite differences for CSI?**
CSI captures the marginal cost of accuracy improvement. A finite difference between
consecutive T values (e.g., T=8 → T=16) gives a practically interpretable "is it worth
doubling T?" signal.

---

## Dependencies (`requirements.txt` preview)

```
torch>=2.0.0
torchvision>=0.15.0
spikingjelly>=0.0.0.0.14
scipy>=1.10.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
ptflops>=0.7.0
tqdm>=4.65.0
```
