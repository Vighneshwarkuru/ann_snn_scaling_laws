# Related Work (Draft Section for Paper)

## 2. Related Work

### 2.1 ANN-SNN Conversion Methods

ANN-SNN conversion has emerged as the most practical approach for deploying deep
SNNs, leveraging pre-trained ANN weights to achieve competitive accuracy without
expensive spike-based backpropagation. The core challenge lies in mapping continuous
ReLU activations to discrete spike rates while minimizing information loss.

**Threshold Balancing.** Diehl et al. [6] introduced model-based and data-based
threshold normalization schemes. Sengupta et al. [34] extended this with SNN spiking
statistics, achieving VGG-16 accuracy of 91.55% on CIFAR-10 but requiring >2000
timesteps. Bu et al. [CVPR 2025] recently proposed local threshold balancing with
channel-wise scaling and delayed evaluation, achieving inference-scale complexity
without quantized ANN retraining. Their framework demonstrates scalability across
classification, segmentation, detection, and video tasks. However, none of these works
derive quantitative models for how conversion efficiency scales with timesteps, depth,
or dataset complexity.

**Soft Reset Neurons.** Han et al. [CVPR 2020] identified that "hard reset" IF neurons
cause information loss during conversion and proposed Residual Membrane Potential
(RMP) neurons with "soft reset." RMP-SNN achieves near loss-less conversion
(93.63% on CIFAR-10, <0.01% loss; 73.09% on ImageNet, 0.4% loss) with 2-8×
fewer timesteps than hard-reset baselines. Hao et al. [AAAI 2023] further analyzed
"unevenness error" (deviation from temporal spike arrival patterns), identifying four
error categories and proving sufficient/necessary conditions for the dominant error
mode. Their method achieves 64.32% on ImageNet with only 10 timesteps.

**Optimal Conversion Theory.** Ding et al. [2021] derived sufficient conditions for
optimal ANN-SNN conversion and proposed Rate Norm Layer (RNL) to replace ReLU,
enabling direct conversion with 8.6× faster inference at 0.265× energy. Wang et al.
proposed negative spike neurons with variance-based threshold optimization for models
using Leaky ReLU. Yan et al. [AAAI 2021] introduced CQ training (Clamped and
Quantized) achieving near-zero conversion loss on VGG architectures (94.16% on
CIFAR-10).

**Our Distinction:** These methods optimize conversion accuracy or latency for
specific configurations. In contrast, we derive *general scaling relationships* governing
how accuracy, energy, and latency scale with timesteps (T), depth (D), and dataset
complexity (S) across 600 experiments. Our framework is method-agnostic—applicable
to standard IF conversion, Han's RMP, Bu's threshold balancing, or any future approach.

---

### 2.2 Direct SNN Training

**Surrogate Gradients.** To overcome the non-differentiability of discrete spikes,
surrogate gradient (SG) methods approximate the gradient through continuous relaxations.
Li et al. [NeurIPS 2021] introduced Differentiable Spike (Dspike), an adaptive SG
family that evolves during training to find optimal shape and smoothness. They achieve
75.4% on CIFAR10-DVS with 10 timesteps using spiking ResNet-18.

**Trainable Neuron Parameters.** Rathi and Roy [2020] proposed DIET-SNN with
learnable membrane leak and firing thresholds optimized via end-to-end backpropagation.
Direct input encoding eliminates spike-train conversion overhead, achieving 69% on
ImageNet with only 5 timesteps and 12× less compute energy than equivalent ANNs.

**Low-Latency Architectures.** Ding et al. [AAAI 2024] introduced Shrinking SNN
(SSNN) with progressive timestep reduction across network stages, achieving 73.63%
on CIFAR10-DVS with 5 average timesteps. Abillama et al. [2025] proposed One-Hot
Multi-Level LIF neurons, demonstrating 2% higher accuracy than conventional VGG-16
SNN on ImageNet while maintaining 20× lower energy than the ANN.

**Our Distinction:** Training-based methods achieve high accuracy at low timesteps but
require expensive gradient computation and SNN-specific training infrastructure.
Conversion remains more practical for deploying *existing* pre-trained models.
Our scaling analysis helps practitioners decide: should they invest in direct SNN
training, or can conversion at appropriate T achieve their accuracy target? The T*
predictor answers this quantitatively.

---

### 2.3 SNN Efficiency Analysis

**Energy Models.** Cao et al. [IJCV 2015] pioneered CNN-to-SNN conversion for
energy-efficient recognition, demonstrating two orders of magnitude energy savings on
neuromorphic hardware. However, their energy claims are hardware-specific and do not
generalize across architectures or timestep settings. Most subsequent works claim
"energy efficiency" without explicit energy models or quantified tradeoffs.

**Latency-Accuracy Tradeoffs.** Several works address the latency problem but report
only fixed-configuration results rather than systematic tradeoff analysis. Han et al.
report "2-8× fewer timesteps" qualitatively; Hao et al. target T=10 specifically;
Rathi and Roy demonstrate T=5. None provide mathematical models predicting
accuracy at arbitrary T or quantifying energy-latency-accuracy tradeoffs across the
full configuration space.

**Our Distinction:** We provide the first *quantitative* energy-latency-accuracy
framework. Our 45nm CMOS model (0.9 pJ/AC, 4.6 pJ/MAC, with Loihi 0.23 pJ/SynOp
extrapolation) enables explicit energy prediction. More importantly, we reveal the
*energy crossover point*—the T beyond which SNN energy advantage diminishes—a
finding absent from all prior work that uniformly claims SNN efficiency.

---

### 2.4 Scaling Laws in Deep Learning

**Neural Scaling Laws.** Kaplan et al. [2020] derived power-law scaling relationships
for language model performance as a function of model size, dataset size, and compute.
Hestness et al. [2017] showed similar scaling across multiple domains. These studies
revealed that performance follows predictable power laws, enabling resource-efficient
planning.

**Scaling Laws for SNNs.** To our knowledge, *no prior work derives empirical scaling
laws for SNN inference complexity.* Existing SNN literature focuses on achieving
specific accuracy targets rather than characterizing how complexity scales across
the multidimensional space of timesteps, depth, and dataset difficulty. This gap
motivates our work.

**Our Contribution:** Inspired by neural scaling law methodology [Kaplan'20], we fit
multi-factor power laws to 600 controlled experiments. Our Complexity Scaling Index
CSI(T,D,S) = α·T^β·D^γ·S^δ quantifies how inference complexity scales with each
factor, revealing their relative importance (β > δ > γ in our experiments). This
is the first such characterization for spiking neural networks, providing theoretical
foundations for neuromorphic deployment analogous to what Kaplan et al. provided for
language model training.

---

## Summary Table: Prior Work vs. Our Approach

| Category | Prior Work Focus | Our Focus |
|----------|-----------------|-----------|
| **Conversion Methods** | Improve accuracy at fixed T | How accuracy scales with T |
| **Training Methods** | Achieve low-T high accuracy | When is training vs conversion optimal |
| **Efficiency Analysis** | "SNNs are efficient" (claim) | When and why SNNs are efficient (model) |
| **Scaling Laws** | None for SNNs | First multi-factor power-law (CSI) |
| **Predictive Models** | None | T* predictor for deployment planning |
| **Statistical Validation** | Rare (1 seed typical) | 3 seeds, t-tests, bootstrap CIs |

---

## Key References to Cite

### Must-cite (direct comparisons):
1. **Bu et al. (CVPR 2025)** - Main competitor, inference-scale conversion
2. **Han et al. (CVPR 2020)** - RMP-SNN, high accuracy baseline
3. **Hao et al. (AAAI 2023)** - Unevenness error, low-latency
4. **Ding et al. (2021)** - Optimal conversion theory
5. **Rathi & Roy (2020)** - DIET-SNN, trainable parameters

### Should-cite (context):
6. **Li et al. (NeurIPS 2021)** - Differentiable Spike (training method)
7. **Ding et al. (AAAI 2024)** - Shrinking SNN
8. **Yan et al. (AAAI 2021)** - CQ training
9. **Cao et al. (IJCV 2015)** - Pioneering CNN→SNN energy work
10. **Abillama et al. (IEEE Access 2025)** - Multi-level LIF

### Background-cite (scaling laws inspiration):
11. **Kaplan et al. (2020)** - Neural scaling laws (language models)
12. **Hestness et al. (2017)** - Scaling across domains
13. **Sengupta et al. (2018)** - Early threshold balancing
14. **Horowitz (2014)** - 45nm CMOS energy model source

---

## Positioning One-Liner (for Introduction)

> "While significant progress has been made in improving ANN-SNN conversion
> accuracy [Bu'25, Han'20, Hao'23] and reducing inference latency [Rathi'20,
> Ding'24], the fundamental question—*how does inference complexity scale
> with timesteps, depth, and dataset complexity?*—remains unanswered. We
> address this gap through the first systematic scaling law study of ANN-SNN
> conversion, deriving predictive models from 600 controlled experiments."
