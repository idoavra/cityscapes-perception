import torch

# --- Paths ---
# Update these to match your local machine
DATA_DIR = r"C:\datasets\Cityspaces\images"
MASK_DIR = r"C:\datasets\Cityspaces\gtFine"
CHECKPOINT_DIR = "checkpoints"

# --- Hardware ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Set to 0 if you are on Windows and get errors, otherwise 4 is good for i5-8400
NUM_WORKERS = 4 

# --- Model Selection ---
# Options: "deeplabv3plus", "unet", "manet"
MODEL_TYPE = "deeplabv3plus"
ENCODER = "efficientnet-b3"
RESUME = False

# --- Training Hyperparameters ---
CLASSES = 19
BATCH_SIZE = 8
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
ALPHA = 0.5        # Balance between CE and Dice loss
PATIENCE = 50     # Early stopping: epochs to wait for improvement

# --- Preprocessing ---
RESIZE = False      # Whether to divide height/width by 2
CACHE = False       # Store dataset in RAM (Turn OFF if you get Out of Memory errors)
DROP_LAST = True