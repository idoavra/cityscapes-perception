import torch
from tqdm import tqdm
import config
from src.dataset import get_cityscapes_loaders
from src.models import get_model
from src.metrics import StreamSegMetrics
from src.utils import load_checkpoint, decode_segmap
import matplotlib.pyplot as plt
import torch.nn as nn

def test():
    # 1. Load Data (We only need the test_loader)
    _, _, test_loader = get_cityscapes_loaders(
        data_dir=config.DATA_DIR,
        mask_dir=config.MASK_DIR,
        batch_size=1, # Keep at 1 for clean evaluation
        resize=config.RESIZE
    )

    # 2. Setup Model
    model = get_model(config.MODEL_TYPE, config.ENCODER, config.CLASSES).to(config.DEVICE)
    model.segmentation_head = nn.Sequential(
        nn.Dropout2d(p=0.1),
        *list(model.segmentation_head.children())
    )
    # 3. Load your best weights
    checkpoint_path = "checkpoints/best_model.pth"
    try:
        load_checkpoint(checkpoint_path, model)
    except FileNotFoundError:
        print(f"Error: Could not find {checkpoint_path}. Did you finish training?")
        return

    # 4. Initialize Metrics
    metrics = StreamSegMetrics(config.CLASSES)
    labels = [
        "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", 
        "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", 
        "truck", "bus", "train", "motorcycle", "bicycle"
    ]

    # 5. Evaluation Loop
    model.eval()
    print(f"Running final evaluation on {len(test_loader)} images...")
    
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            
            # This handles the mask/ignore_index (255) internally in metrics.py!
            metrics.update(y, preds)

    # 6. Get Results
    results = metrics.get_results()
    miou_per_class = results["IoU"]
    overall_miou = results["Overall mIoU"]

    # 7. Print Pretty Table
    print("\n" + "="*30)
    print(f"{'Class':<15} | {'IoU (%)':<10}")
    print("-" * 30)
    for i in range(len(labels)):
        iou_val = miou_per_class[i].item() * 100
        print(f"{labels[i]:<15} | {iou_val:>8.2f}%")
    
    print("-" * 30)
    print(f"{'OVERALL mIoU':<15} | {overall_miou * 100:>8.2f}%")
    print("="*30)

if __name__ == "__main__":
    test()