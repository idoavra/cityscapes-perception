import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def save_checkpoint(state, filename="checkpoints/best_model.pth"):
    """Saves the model and optimizer state."""
    print(f"=> Saving checkpoint to {filename}")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Loads the model and optimizer state."""
    print(f"=> Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint.get('epoch', 0)

def calculate_class_weights(dataloader, num_classes=19):
    """
    Calculates weights based on pixel frequency. 
    Helps the model learn rare classes (poles, signs) better.
    """
    class_names = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
        'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
        'truck', 'bus', 'train', 'motorcycle', 'bicycle'
    ]

    print("Calculating class weights... this may take a moment.")
    counts = torch.zeros(num_classes)
    for _, mask in dataloader:
        mask = mask.flatten()
        # Exclude ignore_index (255)
        valid_mask = mask[mask != 255]
        counts += torch.bincount(valid_mask, minlength=num_classes)
    
    # Weight = 1 / log(1.02 + frequency) 
    # This is the standard "ENet" weighting technique
    probs = counts / counts.sum()
    weights = 1 / torch.log(1.02 + probs)

    print("\n" + "="*30)
    print(f"{'Class Name':<15} | {'Weight':<10}")
    print("-" * 30)
    for i in range(num_classes):
        name = class_names[i] if i < len(class_names) else f"Class {i}"
        print(f"{name:<15} | {weights[i]:.4f}")
    print("="*30 + "\n")
    
    return weights

def decode_segmap(image, num_classes=19):
    """
    Converts class IDs (0-18) into RGB colors for visualization.
    Uses standard Cityscapes colors.
    """
    colors = np.array([
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
        [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
        [0, 80, 100], [0, 0, 230], [119, 11, 32]
    ])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, num_classes):
        idx = image == l
        r[idx] = colors[l, 0]
        g[idx] = colors[l, 1]
        b[idx] = colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

def visualize_prediction(model, dataset, device, epoch, num_samples=3, save_dir="visuals"):
    """
    Takes samples from the dataset and saves a plot of:
    Original Image | Ground Truth | Model Prediction
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    fig, ax = plt.subplots(num_samples, 3, figsize=(15, num_samples * 4))
    
    # Handle case where num_samples is 1 (ax would not be a 2D array)
    if num_samples == 1:
        ax = np.expand_dims(ax, axis=0)

    for i in range(num_samples):
        # Pick a sample (you can change 'i' to 'i*10' to see different images)
        img, mask = dataset[i]
        img_tensor = img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # Denormalize image for display
        img_display = img.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_display = std * img_display + mean
        img_display = np.clip(img_display, 0, 1)

        ax[i, 0].imshow(img_display)
        ax[i, 0].set_title("Original Image")
        ax[i, 1].imshow(decode_segmap(mask.numpy()))
        ax[i, 1].set_title("Ground Truth")
        ax[i, 2].imshow(decode_segmap(pred))
        ax[i, 2].set_title("Prediction")
        
        for j in range(3): 
            ax[i, j].axis('off')

    plt.tight_layout()
    
    # --- The critical changes ---
    save_path = os.path.join(save_dir, f"epoch_{epoch+1}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)  # Closes the figure so it doesn't pop up or stay in RAM
    print(f"=> Visualization saved to {save_path}")


def plot_training(train_miou_list, train_loss_list, val_miou_list, val_loss_list, save_dir="plots"):
        """Generates and saves Loss and mIoU curves."""
        os.makedirs(save_dir, exist_ok=True)
        x = range(1, len(train_miou_list) + 1)

        # -------- Loss Plot --------
        plt.figure(figsize=(10, 5))
        plt.plot(x, train_loss_list, label="Train Loss", color='tab:blue', linewidth=2)
        plt.plot(x, val_loss_list, label="Val Loss", color='tab:red', linestyle='--')
        plt.title("Training & Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, "loss.png"), dpi=300, bbox_inches="tight")
        plt.close() # Free memory

        # -------- mIoU Plot --------
        plt.figure(figsize=(10, 5))
        plt.plot(x, train_miou_list, label="Train mIoU", color='tab:blue', linewidth=2)
        plt.plot(x, val_miou_list, label="Val mIoU", color='tab:red', linestyle='--')
        plt.title("Training & Validation mIoU")
        plt.xlabel("Epochs")
        plt.ylabel("mIoU")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, "mIoU.png"), dpi=300, bbox_inches="tight")
        plt.close() # Free memory