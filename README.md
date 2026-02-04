# 🚗 Cityscapes Semantic Segmentation

**High-performance semantic segmentation for autonomous driving** | Part of a multi-task perception system

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📊 Current Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Val mIoU** | **66.62%** | Experiment 2.2 (640×640 resolution) |
| **Val mIoU (TTA)** | **66.99%** | +0.99% with Test-Time Augmentation |
| **Train mIoU** | 75.03% | Slight overfitting (gap: 8.4%) |
| **Training Time** | ~6-8 hours | 150 epochs on RTX 2060 |

**From baseline 53% → 66.6% val mIoU** (+13.6% improvement!)

---

## 🎯 Features

- ✅ **DeepLabV3+ with EfficientNet-B3** encoder (ImageNet pretrained)
- ✅ **640×640 resolution** for better ASPP context and small object detection
- ✅ **Joint Loss**: Focal Loss + Dice Loss for class imbalance handling
- ✅ **Moderate augmentation** (spatial + photometric transforms)
- ✅ **Test-Time Augmentation** (6x ensemble: 3 scales + h-flip)
- ✅ **Mixed precision training** (AMP) for memory efficiency
- ✅ **Cosine Annealing LR** with warm restarts
- ✅ **Progressive encoder unfreezing** for transfer learning
- ✅ **Comprehensive experiment tracking** in `experiment_log.txt`

---

## 🧪 Experiment Journey

| Exp | Change | Train mIoU | Val mIoU | Gap | Result |
|-----|--------|-----------|----------|-----|--------|
| **Baseline** | - | 61.0% | 53.0% | 8.0% | - |
| 1.1 | Dropout 0.3 | 66.1% | 53.5% | 12.7% | ❌ Worse gap |
| 2.1 | Aggressive aug | 48.4% | 47.3% | 1.1% | ❌ Underfitting |
| **2.1b** | Moderate aug | 70.0% | 58.8% | 11.2% | ✅ +5.8% val |
| **2.2** | Resolution 640×640 | **75.0%** | **66.6%** | 8.4% | ✅ **+13.6% val!** |
| **3.1** | TTA (inference) | 75.0% | **67.0%** | 8.0% | ✅ +1% boost |
| 3.2a | Cosine Annealing | _In progress_ | _TBD_ | _TBD_ | ⏳ Training |

**Key Insights:**
- Resolution increase (512→640) was the biggest win (+7.8% val mIoU)
- Moderate augmentation found the sweet spot (aggressive caused underfitting)
- TTA provides +1% without retraining (especially on small objects: motorcycle +3.67%)

See detailed analysis in [`experiment_log.txt`](experiment_log.txt)

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd Semantic_Segmentation_Cityscapes

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

Download [Cityscapes dataset](https://www.cityscapes-dataset.com/) and update `config.py`:

```python
DATA_DIR = Path(r"C:\datasets\Cityspaces\images")
MASK_DIR = Path(r"C:\datasets\Cityspaces\gtFine")
```

Expected structure:
```
Cityspaces/
├── images/
│   ├── train/
│   └── val/
└── gtFine/
    ├── train/
    └── val/
```

### Training

```bash
# Train from scratch
python train.py

# Resume training (set RESUME=True in config.py)
python train.py
```

### Validation with TTA

```bash
# Test-Time Augmentation validation
python validate_tta.py
```

---

## 📁 Project Structure

```
cityscapes-perception/
├── config.py                   # Hyperparameters & paths
├── train.py                    # Training script
├── validate_tta.py            # TTA validation
├── experiment_log.txt         # Detailed experiment tracking
├── src/
│   ├── dataset.py             # Cityscapes data loader
│   ├── models.py              # DeepLabV3+ / UNet / MANet
│   ├── losses.py              # Focal + Dice loss
│   ├── metrics.py             # mIoU calculation
│   ├── utils.py               # Checkpointing, visualization
│   └── tta.py                 # Test-Time Augmentation
├── checkpoints/               # Model weights
└── plots/                     # Training curves
```

---

## ⚙️ Configuration

Key settings in [`config.py`](config.py):

```python
# Model
MODEL_TYPE = "deeplabv3plus"
ENCODER = "efficientnet-b3"
DROPOUT = 0.2

# Training
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 16
NUM_EPOCHS = 150
LEARNING_RATE = 5e-5

# Loss
DICE_LOSS_WEIGHT = 1.1
FOCAL_LOSS_WEIGHT = 2.0

# Data
RESIZE = False  # Use full 640×640 resolution
CACHE = False   # Set True if you have 32GB+ RAM
```

---

## 🛠️ Hardware Requirements

**Minimum:**
- GPU: NVIDIA GTX 1060 6GB
- RAM: 16GB
- Storage: 50GB (for Cityscapes)

**Tested on:**
- GPU: RTX 2060 6GB
- CPU: Intel i5-8400
- RAM: 32GB
- OS: Windows 11

---

## 📈 Technical Details

### Architecture
- **Backbone**: EfficientNet-B3 (ImageNet pretrained)
- **Decoder**: DeepLabV3+ with ASPP (Atrous Spatial Pyramid Pooling)
- **Input**: 640×640 crops from 2048×1024 Cityscapes images
- **Output**: 19-class pixel-wise predictions

### Loss Function
```
L_total = Focal(γ=2.0, weight=2.0) + Dice(weight=1.1)
```
- **Focal Loss**: Handles class imbalance by focusing on hard examples
- **Dice Loss**: Optimizes IoU directly

### Data Augmentation
```python
# Training (moderate)
- RandomCrop 640×640
- HorizontalFlip (p=0.5)
- ShiftScaleRotate (±10°, 0.9-1.1x scale, p=0.3)
- ColorJitter (brightness, contrast, saturation, hue)
- Light blur (motion/gaussian, p=0.1)

# Validation
- CenterCrop 640×640
- Normalize (ImageNet stats)
```

### Training Strategy
1. **Progressive unfreezing**: Encoder frozen for first 7 epochs
2. **Cosine annealing**: LR restarts every 30 epochs
3. **Mixed precision**: AMP for memory efficiency
4. **Early stopping**: Patience of 50 epochs

---

## 🗺️ Roadmap

### Phase 1: Segmentation ✅
- [x] Baseline DeepLabV3+ (53% mIoU)
- [x] Data augmentation optimization
- [x] Resolution increase to 640×640 (66.6% mIoU)
- [x] Test-Time Augmentation (+1% boost)
- [ ] Cosine annealing LR (in progress)

### Phase 2: Object Detection 📋
- [ ] Integrate YOLOv11 detection head
- [ ] Multi-task learning (shared backbone)
- [ ] Joint training (seg + det)
- [ ] Real-time optimization (30+ FPS target)

### Phase 3: Full Perception System 🔮
- [ ] Depth estimation
- [ ] Lane detection
- [ ] Multi-camera fusion
- [ ] Temporal modeling (video)

**Goal**: Build a Tesla-like autonomous driving perception system 🚗💨

---

## 📚 References

- **DeepLabV3+**: [Encoder-Decoder with Atrous Separable Convolution](https://arxiv.org/abs/1802.02611)
- **EfficientNet**: [Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946)
- **Cityscapes**: [The Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- **Focal Loss**: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

- **Cityscapes team** for the dataset
- **segmentation_models_pytorch** for model implementations
- **Claude Code** for development assistance

---

<p align="center">
  <i>Part of an autonomous driving perception system project</i><br>
  <i>Next: Multi-task learning with object detection</i>
</p>
