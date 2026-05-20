# VECMNet

VECMNet is a text-to-image person re-identification framework based on Variational Noise Modeling (VNM), Consistency-Guided Refinement (CGR), and Cross-Modal Masked Modeling (CMM).

The task is Text-to-Image Person Re-identification (TIReID): given a natural-language pedestrian description, the model retrieves the matching person images from an image gallery. This codebase uses CLIP ViT-B/16 as the vision-language backbone, learns global and local cross-modal representations, and improves robustness under noisy image-text correspondences.

## Overview

The framework contains three main modules:
<img width="4252" height="2250" alt="framework" src="https://github.com/user-attachments/assets/7470e15c-82c0-42b6-b72f-f7c7498734ca" />

1. **VNM: Variational Noise Modeling**

   VNM computes per-sample image-text matching losses at both global and local scales. It then fits a two-component Variational Bayesian Gaussian Mixture Model to the loss distribution, allowing the training pipeline to separate clean image-text pairs from noisy pairs.

2. **CGR: Consistency-Guided Refinement**

   CGR refines the initial pseudo labels produced by VNM using global-local cross-modal consistency. It combines confidence estimation, sample reweighting, distribution consistency, and pseudo-negative consistency regularization to reduce the influence of semantically inconsistent pairs.

3. **CMM: Cross-Modal Masked Modeling**

   CMM randomly masks text tokens and uses image features to help recover the masked words. The masked text features are used as queries, while image features are used as keys and values in a cross-modal Transformer. This encourages fine-grained alignment between image regions and text tokens.

## Repository Structure

```text
VECMNet/
+-- datasets/                  # Dataset loading, preprocessing, and samplers
+-- model/
|   +-- build.py               # Main model definition and forward process
|   +-- clip_model.py          # CLIP image/text encoders
|   +-- CrossEmbeddingLayer_tse.py
|   +-- objectives.py          # TAL, CGR, and CMM losses
+-- processor/
|   +-- processor.py           # Training, inference, and VNM sample division
+-- solver/                    # Optimizer and learning-rate scheduler
+-- utils/                     # Options, logging, metrics, and utility functions
+-- train.py                   # Training entry point
+-- test.py                    # Evaluation entry point
+-- rank.py                    # Text-query Top-K visualization
+-- run_vecm.sh                # Example training script
```

## Requirements

Recommended environment:

```text
Python 3.8+
PyTorch
torchvision
numpy
scipy
scikit-learn
prettytable
easydict
pyyaml
matplotlib
Pillow
tqdm
```

The CLIP pretrained checkpoint is downloaded automatically on first use according to `--pretrain_choice`.

## Supported Datasets

The code supports common text-to-image person re-identification datasets:

```text
CUHK-PEDES
ICFG-PEDES
RSTPReid
```

Set `--root_dir` to the dataset root path. The corresponding dataset reader should provide image paths, text descriptions, and identity labels.

During training, noisy image-text correspondences can be injected or loaded using `--noisy_rate` and `--noisy_file`.

## Training

Edit `run_vecm.sh` to set the dataset path, dataset name, and noise rate, then run:

```bash
sh run_vecm.sh
```

You can also start training directly:

```bash
python train.py \
  --root_dir /path/to/datasets \
  --dataset_name RSTPReid \
  --batch_size 64 \
  --num_epoch 60 \
  --loss_names TAL+sr0.3_tau0.015_margin0.1_n0.2 \
  --noisy_rate 0.2 \
  --noisy_file ./noiseindex/RSTPReid_0.2.npy \
  --img_aug \
  --txt_aug
```

Training logs, saved configs, and checkpoints are stored under:

```text
output_dir/dataset_name/time_name_loss/
```

Common checkpoint files:

```text
best.pth
last.pth
```

## Evaluation

Before evaluation, edit the `sub` variable in `test.py` and set it to the training output directory, for example:

```text
run_logs/RSTPReid/202xxxxx_VECM_TAL+...
```

Then run:

```bash
python test.py
```

The evaluator reports three retrieval settings:

```text
BGE      # global image-text features
TSE      # local enhanced features
BGE+TSE  # fused global and local similarities
```

Metrics:

```text
Rank-1
Rank-5
Rank-10
mAP
mINP
```


## Main Options

### Basic Training Options

```text
--pretrain_choice      CLIP pretrained model, default: ViT-B/16
--img_size             Input image size, default in code: (384, 128)
--stride_size          ViT patch stride, default: 16
--text_length          Maximum text token length, default: 77
--batch_size           Training batch size, default: 64
--test_batch_size      Evaluation batch size, default: 512
--select_ratio         TSE token selection ratio, default: 0.3
--tau                  TAL temperature, default: 0.015
--margin               TAL margin, default: 0.1
```

