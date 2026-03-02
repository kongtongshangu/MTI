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
```

### Prepare Data

We follow the multi-modal corruption protocol used in prior MM-TTA works (15 video corruptions × 6 audio corruptions = 90 combinations).

Step 1: Generate corrupted video/audio data
```bash
# Video corruptions
python data_process/make_c_video.py --corruption gaussian_noise --severity 5 --data-path /path/to/video_val
```

```bash
# Audio corruptions
python data_process/make_c_audio.py --corruption crowd --severity 5 --data-path /path/to/audio_val
```

Step 2: Create JSON files for evaluation
```bash
python data_process/create_video_audio_json.py --video_c_type gaussian_noise --audio_c_type crowd --severity 5 --json_root ./json_csv_files/ks50
```
### Note: 
#### · Remember to change the --clean-path --video-c-path --audio-c-path to adapt your own case. 
#### · You can download the original data from [here](https://drive.google.com/drive/folders/1SWkNwTqI08xbNJgz-YU2TwWHPn5Q4z5b). 
#### · For more details on data preparation, please refer to [READ](https://github.com/XLearning-SCU/2024-ICLR-READ). 

### Prepare Pre-trained Models

The pre-trained models are provided by [READ](https://github.com/XLearning-SCU/2024-ICLR-READ). The pre-trained model for KS50 and VGGSound are [ks50](https://drive.google.com/file/d/1m38uCAfwL--RP6rWtOvGee4i2SfAzbjl/view) and [vgg_65.5](https://uc7264f246f3729c80858ed9e281.dl.dropboxusercontent.com/cd/0/get/C74gv1WsG61OcyRgamnuyrhEYMLXejmdUauksAeDiFfHXtbSPzSOWuyBDwZ3VHNWwsr0H81g52rFvryBDxr1Tj0YlvZvtKbRMhyB-s1fZr2DiYvVHl6t2VAtGgqR72oIsyIjOflJP-nOHk4D7bEe9jIr/file?dl=1#), respectively.

```bash
mkdir -p pretrained
# Put your checkpoint here:
# pretrained/cav_mae_ks50.pth
```

## Run Test-Time Adaptation
Example: Kinetics50-MC, both modalities corrupted
CUDA_VISIBLE_DEVICES=0 python run.py --dataset ks50 --tta-method OURS --pretrain_path ./pretrained/cav_mae_ks50.pth --corruption-modality both --audio_c_type crowd

## Acknowledgement
- PTA code is heavily used. [official](https://github.com/MPI-Lab/PTA)
- READ code is heavily used. [official](https://github.com/XLearning-SCU/2024-ICLR-READ)
