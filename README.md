# Semantic Segmentation on Cityscapes

PyTorch implementation of semantic segmentation using modern architectures on the Cityscapes dataset for autonomous driving scene understanding.

## Features

- **Multiple Architectures**: DeepLabV3+, U-Net, MANet with various encoder backbones
- **Advanced Training Techniques**:
  - Mixed precision training (AMP) for faster training
  - Gradient accumulation for larger effective batch sizes
  - Progressive encoder unfreezing for transfer learning
  - Class weight balancing for handling imbalanced classes
- **Custom Joint Loss**: Combines Focal Loss + Dice Loss for better segmentation
- **Data Augmentation**: Albumentations pipeline with geometric and photometric transforms
- **Resume Training**: Checkpoint system with full training history
- **Visualization**: Automatic saving of predictions and training curves

## Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 6GB+ VRAM for training

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Semantic_Segmentation_Cityscapes
```

2. Create a virtual environment:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Download the Cityscapes dataset from [cityscapes-dataset.com](https://www.cityscapes-dataset.com/)
   - You need: `leftImg8bit_trainvaltest.zip` (images)
   - And: `gtFine_trainvaltest.zip` (labels)

2. Extract the dataset and update paths in `config.py`:
```python
DATA_DIR = r"C:\path\to\cityscapes\leftImg8bit"
MASK_DIR = r"C:\path\to\cityscapes\gtFine"
```

3. The expected directory structure:
```
cityscapes/
├── leftImg8bit/
│   ├── train/
│   ├── val/
│   └── test/
└── gtFine/
    ├── train/
    ├── val/
    └── test/
```

## Usage

### Training

Start training with default settings:
```bash
python train.py
```

The training script will:
- Automatically calculate class weights
- Save best model to `checkpoints/best_model.pth`
- Generate training curves in `plots/`
- Create prediction visualizations in `visuals/`

### Evaluation

Test your trained model:
```bash
python test.py
```

This will load the best checkpoint and evaluate on the test set, printing per-class IoU scores.

### Configuration

Adjust hyperparameters in `config.py`:

```python
# Model Selection
MODEL_TYPE = "deeplabv3plus"  # Options: "unet", "deeplabv3plus", "manet"
ENCODER = "efficientnet-b3"   # Any timm encoder
DROPOUT = 0.2                  # Dropout for segmentation head

# Training
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
NUM_EPOCHS = 100
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

# Data
RESIZE = True  # Resize images to half resolution for faster training
CACHE = False  # Cache dataset in RAM (requires ~20GB RAM)

# Loss
DICE_LOSS_WEIGHT = 1.1  # Balance between Focal and Dice loss
```

## Project Structure

```
Semantic_Segmentation_Cityscapes/
├── config.py              # Configuration file
├── train.py               # Training script
├── test.py                # Evaluation script
├── requirements.txt       # Dependencies
├── src/
│   ├── dataset.py         # Cityscapes dataset loader
│   ├── models.py          # Model factory (U-Net, DeepLabV3+, MANet)
│   ├── losses.py          # Custom loss functions (Focal + Dice)
│   ├── metrics.py         # Metrics computation (mIoU)
│   └── utils.py           # Utilities (checkpoints, visualization)
├── checkpoints/           # Model checkpoints
├── plots/                 # Training curves (loss, mIoU)
└── visuals/               # Prediction visualizations
```

## Results

| Model | Encoder | Input Size | mIoU | Notes |
|-------|---------|------------|------|-------|
| DeepLabV3+ | EfficientNet-B3 | 512×512 | TBD% | With progressive unfreezing |

> Update this table with your results after training

## Training Tips

1. **Overfitting?** Uncomment `A.GaussNoise(p=0.1)` in `src/dataset.py`
2. **Out of memory?** Reduce `BATCH_SIZE` or enable `RESIZE = True`
3. **Faster training?** Enable `CACHE = True` if you have 20GB+ RAM
4. **Different encoder?** Try `"resnet50"`, `"efficientnet-b0"`, or any [timm encoder](https://github.com/rwightman/pytorch-image-models)

## Technical Details

### Progressive Encoder Unfreezing
The encoder starts frozen and unfreezes after 5% of epochs (default: epoch 5 for 100 epochs). This allows the decoder head to learn first before fine-tuning the pretrained encoder.

### Joint Loss Function
Combines Focal Loss (handles class imbalance) with Dice Loss (optimizes IoU directly):
```
Loss = FocalLoss + α × DiceLoss
```

### Data Augmentation
- **Training**: Random crop (512×512), horizontal flip, brightness/contrast jitter
- **Validation**: Center crop (512×512) for consistent evaluation

## License

MIT License - feel free to use this code for your projects.

## Acknowledgments

- [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) for model implementations
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/) for the autonomous driving benchmark
