import torch
from tqdm import tqdm
import os

# Import your custom modules
import config
from src.dataset import get_cityscapes_loaders
from src.models import get_model
from src.losses import JointLoss
from src.metrics import StreamSegMetrics
from torch.amp import autocast, GradScaler # Added for speed
# Added plot_training to the imports
from src.utils import calculate_class_weights, save_checkpoint, visualize_prediction, plot_training
import torch.nn as nn

def main():
    # 1. Prepare Data
    # Added drop_last=True here to prevent the "1 image batch" error
    train_loader, val_loader, _ = get_cityscapes_loaders(
        data_dir=config.DATA_DIR,
        mask_dir=config.MASK_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        resize=config.RESIZE,
        cache=config.CACHE,
        drop_last=config.DROP_LAST # <--- Crucial fix for BatchNorm
    )

    # 2. Setup Model
    model = get_model(
        model_name="deeplabv3plus", 
        encoder_name="efficientnet-b3", 
        classes=config.CLASSES
    ).to(config.DEVICE)

    dropout_layer = nn.Dropout2d(p=0.2)

# 2. Re-wrap the segmentation_head
# This places Dropout BEFORE the Conv2d layer
    model.segmentation_head = nn.Sequential(
    dropout_layer,
    *list(model.segmentation_head.children())
)

    print(model.segmentation_head)

    # 3. Handle Class Imbalance
    weights = calculate_class_weights(train_loader, num_classes=config.CLASSES).to(config.DEVICE)
    
    # 4. Loss, Optimizer, and Metrics
    criterion = JointLoss(num_classes=config.CLASSES, alpha=config.ALPHA, weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler() # <--- AMP Scaler: Saves VRAM & Speeds up math
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    metrics = StreamSegMetrics(config.CLASSES)

    # 5. Training Variables
    checkpoint_path = "checkpoints/best_model.pth"
    best_miou = 0.0
    early_stop_counter = 0
    start_epoch = 0
    unfreeze = False
    
    # Persistent history (prevents plots from restarting at 0 when resuming)
    history = {
        "train_loss": [],
        "train_miou": [],
        "val_loss": [],
        "val_miou": []
    }

    # --- RESUME LOGIC ---
    if os.path.exists(checkpoint_path) and config.RESUME == True:
        print(f"==> Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Load scaler and history so training remains seamless
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if 'history' in checkpoint:
            history = checkpoint['history']
            
        start_epoch = checkpoint['epoch'] + 1
        best_miou = checkpoint['miou']
        print(f"==> Resuming at Epoch {start_epoch} with Best mIoU: {best_miou:.4f}")
    else:
        print("No checkpoint has found, start")

    # Ensure encoder starts frozen (if not already handled by unfreeze logic)
    if start_epoch < config.NUM_EPOCHS // 20:
        for p in model.encoder.parameters():
            p.requires_grad = False

    # 6. Training Loop
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        
        # --- Unfreeze Logic ---
        if epoch >= config.NUM_EPOCHS // 20 and not unfreeze:
            print("=> Unfreezing Encoder...")
            for p in model.encoder.parameters():
                p.requires_grad = True
            
            # Check if encoder params are already in the optimizer
            existing_params = set()
            for group in optimizer.param_groups:
                for p in group['params']:
                    existing_params.add(p)
            
            encoder_params = list(model.encoder.parameters())
            if encoder_params[0] not in existing_params:
                optimizer.add_param_group({'params': encoder_params, 'lr': 5e-6})
                print("=> Encoder parameters added to optimizer group.")
            else:
                print("=> Encoder parameters already present in optimizer. Skipping add.")
            unfreeze = True

        # --- Training Phase ---
        model.train()
        metrics.reset()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]")
        
        for i, (x, y) in enumerate(pbar):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            
            # Use autocast for the forward pass (16-bit math)
            with autocast(device_type='cuda'):
                logits = model(x)
                loss = criterion(logits, y)
                loss = loss / config.ACCUM_STEPS

            # Scaler handles the backward pass and step to prevent underflow
            scaler.scale(loss).backward()
            if (i + 1) % config.ACCUM_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            metrics.update(y, preds)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_stats = metrics.get_results()
        
        # --- Validation Phase ---
        model.eval()
        metrics.reset()
        val_loss = 0.0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(config.DEVICE), y.to(config.DEVICE)
                
                with autocast(device_type='cuda'): # Faster validation
                    logits = model(x)
                    loss = criterion(logits, y)
                
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                metrics.update(y, preds)

        val_stats = metrics.get_results()
        current_miou = val_stats["Overall mIoU"]
        
        # Update history
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        history["train_loss"].append(avg_train_loss)
        history["train_miou"].append(train_stats['Overall mIoU'])
        history["val_loss"].append(avg_val_loss)
        history["val_miou"].append(current_miou)

        # Logging
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Train mIoU: {train_stats['Overall mIoU']:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f} | Val mIoU: {current_miou:.4f}")

        # NEW: Per-Class mIoU Printing
        class_names = [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
            'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
            'truck', 'bus', 'train', 'motorcycle', 'bicycle'
        ]
        
        if epoch % 10 == 0:
            # Access the per-class IoU from your metrics object
            # Most StreamSegMetrics return this as a dictionary or list under 'Class IoU'
            if "Class IoU" in val_stats:
                print(f"{'Class Name':<15} | {'IoU Score':<10}")
                print("-" * 30)
                class_ious = val_stats["Class IoU"]
                for i, iou in enumerate(class_ious):
                    name = class_names[i] if i < len(class_names) else f"Class {i}"
                    # Highlight low-performing classes in red-ish text (optional console logic)
                    print(f"{name:<15} | {iou:.4f}")
            
            print("="*40 + "\n")

        # Scheduler & Early Stopping
        scheduler.step(current_miou)
        
        if current_miou > best_miou:
            best_miou = current_miou
            early_stop_counter = 0
            save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(), # Save AMP state
                'history': history,                      # Save history
                'epoch': epoch,
                'miou': best_miou
            }, filename="checkpoints/best_model.pth")
            
            print(f"New Best mIoU: {best_miou:.4f} - Model Saved!")
            
            # Plot using the full history
            
            plot_training(
                history["train_miou"], 
                history["train_loss"], 
                history["val_miou"], 
                history["val_loss"], 
                save_dir="plots"
            )
            print("History Saved!")
        else:
            early_stop_counter += 1
            print(f"No improvement for {early_stop_counter}/{config.PATIENCE} epochs.")

        if early_stop_counter >= config.PATIENCE:
            print("Early Stopping Triggered. Ending Training.")
            break
            
        if (epoch + 1) % 10 == 0:
            visualize_prediction(model, val_loader.dataset, config.DEVICE, epoch)

if __name__ == "__main__":
    main()