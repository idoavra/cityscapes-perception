# Multi-Task Perception: Semantic Segmentation + Object Detection on Cityscapes

A multi-task deep learning model that jointly performs **semantic segmentation** (19 classes) and **object detection** (8 classes) on the Cityscapes urban driving dataset. Built on a shared EfficientNet-B3 backbone with a DeepLabV3+ segmentation branch and a YOLO11s detection branch. Designed with ROS2/Gazebo integration in mind.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

## Architecture

| Metric | Value | Notes |
|--------|-------|-------|
| **Val mIoU** | **66.62%** | Experiment 2.2 (640×640 resolution) |
| **Val mIoU (TTA)** | **67.61%** | +0.99% with Test-Time Augmentation |
| **Train mIoU** | 75.03% | Slight overfitting (gap: 8.4%) |
| **Training Time** | ~6-8 hours | 150 epochs on RTX 2060 |

**From baseline 53% → 66.6% val mIoU** (+13.6% improvement!)

| Component    | Details                                        |
|--------------|------------------------------------------------|
| Backbone     | EfficientNet-B3, ImageNet pretrained, frozen   |
| Seg branch   | DeepLabV3+ decoder, pretrained to 66.6% mIoU  |
| Det branch   | LightFPN or BiFPN → YOLO11s Detect head       |
| Input size   | 640×640                                        |
| Hardware     | RTX 2060 6GB, i5-8400, Batch 4, grad accum 4  |
| Dataset      | Cityscapes — 2975 train / 500 val              |

**Segmentation classes (19):** road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle

**Detection classes (8):** person, rider, car, truck, bus, train, motorcycle, bicycle

---

## Best Results (Run 4 — Light FPN, no Phase 2)

| Split | mAP@0.5 | mIoU  |
|-------|---------|-------|
| Val   | 0.634   | 0.656 |
| Test  | 0.577   | 0.632 |

Per-class AP:

| Class      | AP    | vs Baseline |
|------------|-------|-------------|
| person     | 0.688 | +0.3pp      |
| rider      | 0.687 | -8.7pp      |
| car        | 0.773 | +1.9pp      |
| truck      | 0.546 | +12.0pp     |
| bus        | 0.785 | +21.5pp     |
| train      | 0.424 | +27.5pp     |
| motorcycle | 0.428 | -7.3pp      |
| bicycle    | 0.649 | -0.2pp      |

---

## Setup

### Requirements

```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install ultralytics
pip install albumentations
pip install efficientnet-pytorch
```

### Dataset

Download [Cityscapes](https://www.cityscapes-dataset.com/) and update `config.py`:

```python
DATA_DIR = Path("path/to/cityscapes/images")
MASK_DIR = Path("path/to/cityscapes/gtFine")
```

---

## Training

### Step 1 — Segmentation baseline (checkpoint provided in `networks/`)

```bash
python train.py
```

Trains DeepLabV3+ segmentation only. Produces `checkpoints/best_model.pth` (~66.6% mIoU).

### Step 2 — Multi-task training

```bash
python MultiHead_train.py
```

Trains FPN + YOLO detection head for 110 epochs with backbone and seg decoder frozen.

Key settings in `config.py`:

```python
PHASE1_EPOCHS = 110   # Full training duration — Phase 2 eliminated
PHASE2_EPOCHS = 0
BiFPN         = False # True to use BiFPN (warning: overfits at this data scale)
```

---

## Inference

```bash
# Single image with visualization
python inference.py --image path/to/image.jpg --show

# Directory of images (saves output alongside originals)
python inference.py --dir path/to/images/

# Quantitative evaluation on test set
python inference.py --test

# Specific checkpoint
python inference.py --test --checkpoint networks/run_4/multitask_best.pth
```

### ROS2 Integration

```python
from inference import MultiTaskPredictor

predictor = MultiTaskPredictor("networks/run_4/multitask_best.pth")
seg_mask, boxes_xyxy, labels, scores = predictor.predict(image_bgr)

# seg_mask   : (H, W) np.uint8   — Cityscapes train IDs 0-18
# boxes_xyxy : (N, 4) np.float32 — pixel coords in original image space
# labels     : (N,)   np.int32   — class IDs 0-7
# scores     : (N,)   np.float32 — confidence [0, 1]
```

---

## Ablation Study

Five training runs were conducted to isolate the effect of each design choice.

| Run | FPN   | Scheduler         | Mosaic | Phase 2     | Val mAP | Val mIoU |
|-----|-------|-------------------|--------|-------------|---------|----------|
| 1   | Light | WarmRestarts      | No     | Yes (60 ep) | 0.600   | 0.663    |
| 2   | Light | CosineAnnealingLR | Yes    | Yes (early) | 0.553   | 0.662    |
| 3   | BiFPN | WarmRestarts      | Yes    | Yes (early) | 0.382   | 0.675    |
| 4   | Light | CosineAnnealingLR | Yes    | No          | 0.634   | 0.656    |
| 5   | BiFPN | CosineAnnealingLR | Yes    | No          | 0.672   | 0.653    |

Run 5 has a higher val mAP but overfits: its test mAP is 0.309 vs Run 4's 0.577.

### Key Findings

#### 1. Phase 2 is architecturally harmful

The original two-phase design had Phase 2 unfreezing the seg decoder for joint fine-tuning. In every run with Phase 2 active, detection regressed from its Phase 1 peak (Run 1: 0.64→0.60, Run 2: stopped early, Run 3: 0.38 final). Two structural causes:

- **Optimizer state reset**: a new optimizer is created at the phase transition, discarding all Phase 1 momentum state built up over 50 epochs.
- **Competing gradients**: unfreezing the decoder introduces gradients that pull shared backbone features away from the detection-optimized configuration, with no new information available (backbone stays frozen in both phases).

Eliminating Phase 2 (Run 4) immediately improved mAP from 0.60 → 0.634 and allowed uninterrupted training for the full 110 epochs.

#### 2. CosineAnnealingWarmRestarts prevents convergence

The sawtooth LR pattern (T0=10/20) periodically resets learning rate to its maximum, ejecting the model from the minimum it was approaching. Replacing with `CosineAnnealingLR` (single smooth decay over the full phase) improved per-class quality. `bus` AP improved by +21pp (Run 1→Run 4) and `truck` by +12pp.

#### 3. Mosaic augmentation dramatically improves rare classes

Standard 640×640 random crops from 2048×1024 Cityscapes images frequently contain zero rare-class instances. Mosaic stitches 4 random crops per sample, increasing effective exposure ~4×. `train` AP improved from 0.149 → 0.424 (+27.5pp), `bus` from 0.570 → 0.785 (+21.5pp). Trade-off: `rider` and `motorcycle` regressed slightly, as they rely on scene context (e.g., rider-on-bicycle) that gets lost in 320×320 mosaic tiles.

#### 4. BiFPN overfits at this dataset scale

BiFPN achieves the highest val mAP (0.672) but has a massive val→test gap (0.672→0.309), compared to Light FPN's gap of 0.634→0.577. BiFPN's extra bottom-up pathway increases parameter count without adding training data, causing the model to memorize the 400-image val distribution. Light FPN's constrained capacity acts as implicit regularization. BiFPN could be viable with dropout on the FPN layers or a larger dataset.

#### 5. COCO pretrained YOLO initialization

6 of the 8 Cityscapes detection classes exist in COCO. The YOLO11s head starts at mAP=0.49 before any Cityscapes training. All improvements above this baseline represent genuine adaptation to the Cityscapes distribution and box statistics.

---

## Project Structure

```
├── MultiHead_train.py      # Multi-task training script
├── train.py                # Segmentation-only baseline (do not modify)
├── inference.py            # Inference + ROS2-ready predictor class
├── config.py               # All hyperparameters
├── compute_map.py          # mAP@0.5 computation (fixed for Cityscapes classes)
├── src/
│   ├── multitask_model.py  # Model definition (EfficientNet + FPN/BiFPN + YOLO)
│   ├── dataset.py          # Cityscapes dataloader with mosaic augmentation
│   ├── losses.py           # Det_Seg_Loss with batch_size normalization fix
│   └── metrics.py          # StreamSegMetrics (confusion matrix mIoU)
├── networks/               # Saved checkpoints per run
│   ├── run_4/multitask_best.pth    # Best generalizing model
│   └── run_5/multitask_best.pth    # Higher val mAP, overfits
└── EXPERIMENT_LOG.md       # Full ablation log with per-run analysis
```

---

## References

- **DeepLabV3+**: [Encoder-Decoder with Atrous Separable Convolution](https://arxiv.org/abs/1802.02611)
- **EfficientNet**: [Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946)
- **BiFPN**: [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)
- **YOLO11**: [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- **Cityscapes**: [The Cityscapes Dataset for Semantic Urban Scene Understanding](https://www.cityscapes-dataset.com/)
- **Focal Loss**: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
