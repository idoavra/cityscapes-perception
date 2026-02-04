import torch
from pathlib import Path
import sys

# --- Paths ---
# Update these to match your local machine
DATA_DIR = Path(r"C:\datasets\Cityspaces\images")
MASK_DIR = Path(r"C:\datasets\Cityspaces\gtFine")
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

# Create checkpoint directory if needed
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Convert back to strings for compatibility
DATA_DIR = str(DATA_DIR)
MASK_DIR = str(MASK_DIR)
CHECKPOINT_DIR = str(CHECKPOINT_DIR)

# --- Hardware ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("⚠️  WARNING: CUDA not available. Training on CPU will be very slow!")

# Set to 0 if you are on Windows and get errors, otherwise 4 is good for i5-8400
NUM_WORKERS = 5 

# --- Model Selection ---
# Options: "deeplabv3plus", "unet", "manet"
MODEL_TYPE = "deeplabv3plus"
ENCODER = "efficientnet-b3"
DROPOUT = 0.2  # Dropout for segmentation head (0.0 = disabled)
RESUME = True  # Experiment 3.2a: Fresh training with Cosine Annealing LR

# --- Training Hyperparameters ---
CLASSES = 19
BATCH_SIZE = 4
NUM_EPOCHS = 150
LEARNING_RATE = 5e-5
DICE_LOSS_WEIGHT = 1.1  # Balance between Focal and Dice loss
PATIENCE = 50           # Early stopping: epochs to wait for improvement
GRADIENT_ACCUMULATION_STEPS = 4  # Accumulate gradients over N batches
FOCAL_LOSS_WEIGHT = 2.0

# --- Preprocessing ---
RESIZE = False      # Whether to divide height/width by 2
CACHE = False       # Store dataset in RAM (Turn OFF if you get Out of Memory errors)
DROP_LAST = True

# Validate configuration
assert BATCH_SIZE > 0, "BATCH_SIZE must be positive"
assert NUM_EPOCHS > 0, "NUM_EPOCHS must be positive"
assert LEARNING_RATE > 0, "LEARNING_RATE must be positive"
assert 0 <= DROPOUT < 1, "DROPOUT must be in range [0, 1)"
assert GRADIENT_ACCUMULATION_STEPS > 0, "GRADIENT_ACCUMULATION_STEPS must be positive"