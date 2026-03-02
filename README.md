# Modulate-Then-Integrate: Redefining Instance Features for Multi-Modal Test-Time Adaptation (MTI)

<p align="center">
  <img src="./Framework.png" width="95%">
</p>

## Overview

Multi-modal test-time adaptation (MM-TTA) aims to mitigate distribution shifts by exploiting multi-modal information without accessing source data. Recent advances in this field primarily rely on attention-guided fusion, which is effective under unimodal shifts but becomes fundamentally limited under simultaneous multi-modal shifts. This limitation stems from the collapse of attention discrimination in the absence of a relatively reliable modality, leading to degenerate fusion representations. To address this, we propose Modulate-Then-Integrate (MTI), a novel MM-TTA method that focuses on instance-level feature modulation before multi-modal fusion. Specifically, MTI consists of two key components: Instance-Aware Mixture-of-Experts Adapter (IAMA) and Stratified Entropy Modulation (SEM). IAMA maintains a set of expert adapters and generates instance-specific expert weights through a lightweight routing network based on each instance’s contextual features, enabling adaptive instance-level feature modulation. The modulated features are then integrated across modalities in the fusion module to produce the final predictions. SEM complements IAMA by stratifying test samples according to their reliability and applying stratum-dependent entropy modulation, providing a stable optimization signal that guides the adaptive modulation. Extensive experiments demonstrate that MTI  outperforms state-of-the-art methods, achieving gains of 3.4\% and 2.0\% on VGGSound-MC and Kinetics50-MC, respectively.

---

> 🔥 Tunable modules: IAMA (router + experts + gating)  
> ❄️ Frozen modules: encoders, fusion layer, and classifier (default setting)

---

## Key Methods

- **Instance-Aware Mixture-of-Experts Adapter (IAMA)**
  - Maintains an expert pool for each modality and produces instance-specific expert weights for token modulation.
- **Stratified Entropy Modulation (SEM)**
  - Stratifies samples by reliability and applies stratum-specific entropy objectives to stabilize test-time updates.

---

## Getting Started

### Requirements

We recommend the following environment (you may adjust based on your setup):

- python >= 3.8
- torch >= 1.13
- torchvision >= 0.14
- torchaudio >= 0.13
- timm
- numpy, tqdm, scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
