import torch
from pathlib import Path
import sys

# ==================== SHARED: Paths & Hardware ====================
# Used by both train.py (segmentation) and MultiHead_train.py (multi-task)

# Paths
DATA_DIR = Path(r"C:\datasets\Cityspaces\images")
MASK_DIR = Path(r"C:\datasets\Cityspaces\gtFine")
BBOX_DIR = MASK_DIR  # Cityscapes annotations (same dir for multi-task)
CHECKPOINT_DIR = Path("checkpoints")

# Validate critical paths
if not DATA_DIR.exists():
    print(f"❌ ERROR: Data directory not found: {DATA_DIR}")
    print("   Please update DATA_DIR in config.py to point to your Cityscapes images")
    print("   Expected structure: DATA_DIR/train/, DATA_DIR/val/")
    sys.exit(1)

if not MASK_DIR.exists():
    print(f"❌ ERROR: Mask directory not found: {MASK_DIR}")
    print("   Please update MASK_DIR in config.py to point to your Cityscapes labels")
    print("   Expected structure: MASK_DIR/train/, MASK_DIR/val/")
    sys.exit(1)

CHECKPOINT_DIR.mkdir(exist_ok=True)

# Convert to strings for compatibility
DATA_DIR = str(DATA_DIR)
MASK_DIR = str(MASK_DIR)
BBOX_DIR = str(BBOX_DIR)
CHECKPOINT_DIR = str(CHECKPOINT_DIR)

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("⚠️  WARNING: CUDA not available. Training on CPU will be very slow!")

NUM_WORKERS = 5  # Set to 0 on Windows if errors occur
CLASSES = 19
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
RESUME = False

# Preprocessing
RESIZE = False
CACHE = False
DROP_LAST = True


# ==================== SEGMENTATION-ONLY: train.py ====================
# Hyperparameters specific to single-task segmentation training

MODEL_TYPE = "deeplabv3plus"
ENCODER = "efficientnet-b3"
DROPOUT = 0.2

NUM_EPOCHS = 150
LEARNING_RATE = 5e-5
PATIENCE = 50

DICE_LOSS_WEIGHT = 1.1
FOCAL_LOSS_WEIGHT = 2.0


# ==================== MULTI-TASK: MultiHead_train.py ====================
# Hyperparameters for multi-task (segmentation + detection) training

# Checkpoints
SEG_CHECKPOINT = "checkpoints/best_model.pth"  # Pretrained segmentation (66.6% mIoU)
MULTITASK_CHECKPOINT = "checkpoints/Multitask/multitask_best.pth"

# Training phases
PHASE1_EPOCHS = 110  # FPN+YOLO only — Phase 2 eliminated (see EXPERIMENT_LOG.md)
PHASE2_EPOCHS = 0
TOTAL_EPOCHS_MULTITASK = PHASE1_EPOCHS

# Multi-task loss weights (adjust if one task dominates)
LAMBDA_SEG = 1.0
LAMBDA_DET = 1.0

# FPN Structure
BiFPN = False

# Phase 1: Train FPN + YOLO only
PHASE1_LR = 5e-4
PHASE1_SCHEDULER_T0 = 10

# Phase 2: Differential learning rates
PHASE2_BACKBONE_LR = 1e-6    # ImageNet pretrained - minimal updates
PHASE2_DECODER_LR = 5e-6     # Trained segmentation - small adjustments
PHASE2_DETECTION_LR = 5e-4   # Active training for FPN + YOLO
PHASE2_SCHEDULER_T0 = 20


# ==================== Validation ====================
assert BATCH_SIZE > 0, "BATCH_SIZE must be positive"
assert NUM_EPOCHS > 0, "NUM_EPOCHS must be positive"
assert LEARNING_RATE > 0, "LEARNING_RATE must be positive"
assert 0 <= DROPOUT < 1, "DROPOUT must be in range [0, 1)"
assert GRADIENT_ACCUMULATION_STEPS > 0, "GRADIENT_ACCUMULATION_STEPS must be positive"
assert PHASE1_EPOCHS > 0, "PHASE1_EPOCHS must be positive"
assert PHASE2_EPOCHS >= 0, "PHASE2_EPOCHS must be non-negative"
assert LAMBDA_SEG > 0 and LAMBDA_DET > 0, "Loss weights must be positive"