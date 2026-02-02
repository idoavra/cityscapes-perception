"""
Validation script with Test-Time Augmentation (TTA)
Evaluates the best model checkpoint using TTA and compares with baseline.
"""
import matplotlib
matplotlib.use('Agg')

import torch
from torch.amp import autocast
from tqdm import tqdm
import config
from src.models import get_model
from src.metrics import StreamSegMetrics
from src.tta import create_tta_model
from src.dataset import get_cityscapes_loaders



def validate(model, val_loader, device, use_tta=False, tta_config='moderate'):
    """
    Run validation with or without TTA.

    Args:
        model: PyTorch model
        val_loader: Validation data loader
        device: Device to run on
        use_tta: Whether to use TTA
        tta_config: TTA configuration ('light', 'moderate', 'aggressive')

    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    metrics = StreamSegMetrics(n_classes=config.CLASSES)

    if use_tta:
        print(f"\n{'='*60}")
        print(f"Running validation WITH TTA ({tta_config} configuration)")
        print(f"{'='*60}")
        tta_model = create_tta_model(model, tta_config=tta_config, device=device)
    else:
        print(f"\n{'='*60}")
        print(f"Running validation WITHOUT TTA (baseline)")
        print(f"{'='*60}")

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            with autocast(device_type='cuda'):
                if use_tta:
                    logits = tta_model(x)
                else:
                    logits = model(x)

            preds = torch.argmax(logits, dim=1)
            metrics.update(y, preds)

            # Update progress bar with current mIoU
            current_stats = metrics.get_results()
            pbar.set_postfix({"mIoU": f"{current_stats['Overall mIoU']:.4f}"})

    val_stats = metrics.get_results()
    return val_stats


def print_comparison(baseline_stats, tta_stats):
    """
    Print a comparison table between baseline and TTA results.

    Args:
        baseline_stats: Metrics from baseline validation
        tta_stats: Metrics from TTA validation
    """
    baseline_miou = baseline_stats['Overall mIoU']
    tta_miou = tta_stats['Overall mIoU']
    improvement = tta_miou - baseline_miou
    gap_reduction = improvement  # Since train mIoU stays the same

    print(f"\n{'='*70}")
    print(f"{'VALIDATION RESULTS COMPARISON':^70}")
    print(f"{'='*70}")
    print(f"{'Metric':<30} | {'Baseline':<15} | {'TTA':<15} | {'Change':<10}")
    print(f"{'-'*70}")
    print(f"{'Val mIoU':<30} | {baseline_miou:>14.2%} | {tta_miou:>14.2%} | {improvement:>+9.2%}")
    print(f"{'='*70}")

    if improvement > 0:
        print(f"\n✅ SUCCESS! TTA improved validation mIoU by {improvement:.2%}")
        print(f"   This reduces the overfitting gap by {gap_reduction:.2%}")
    else:
        print(f"\n❌ TTA did not improve performance (change: {improvement:.2%})")

    print(f"\nPer-class mIoU comparison:")
    print(f"{'-'*70}")
    print(f"{'Class':<20} | {'Baseline':<12} | {'TTA':<12} | {'Change':<10}")
    print(f"{'-'*70}")

    class_names = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
        'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
        'truck', 'bus', 'train', 'motorcycle', 'bicycle'
    ]

    for i in range(len(class_names)):
        baseline_class_miou = baseline_stats['Class IoU'][i]
        tta_class_miou = tta_stats['Class IoU'][i]
        class_change = tta_class_miou - baseline_class_miou

        # Highlight classes with significant improvement (>1%)
        marker = "📈" if class_change > 0.01 else ""

        print(f"{class_names[i]:<20} | {baseline_class_miou:>11.2%} | {tta_class_miou:>11.2%} | {class_change:>+9.2%} {marker}")

    print(f"{'='*70}\n")


def main():
    """
    Main function to run TTA validation.
    """
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # Load data
    print("\nLoading validation dataset...")
    _, val_loader, _ = get_cityscapes_loaders(
        data_dir=config.DATA_DIR,
        mask_dir=config.MASK_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        resize=config.RESIZE,
        cache=config.CACHE,
        drop_last=config.DROP_LAST # <--- Crucial fix for BatchNorm
    )
    print(f"Validation batches: {len(val_loader)}")

    # Load model
    print("\nLoading model...")
    model = get_model(
        model_name=config.MODEL_TYPE,
        encoder_name=config.ENCODER,
        classes=config.CLASSES,
        dropout=config.DROPOUT
    ).to(device)

    # Load best checkpoint
    checkpoint_path = f"{config.CHECKPOINT_DIR}/best_model.pth"
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with mIoU: {checkpoint['miou']:.4f}")

    # Run baseline validation (without TTA)
    baseline_stats = validate(model, val_loader, device, use_tta=False)

    # Run TTA validation
    # Try 'light' first for quick results, then 'moderate' for better performance
    tta_config = 'moderate'  # Options: 'light', 'moderate', 'aggressive'
    tta_stats = validate(model, val_loader, device, use_tta=True, tta_config=tta_config)

    # Print comparison
    print_comparison(baseline_stats, tta_stats)

    # Save results to file
    results_file = "tta_results.txt"
    with open(results_file, 'w') as f:
        f.write(f"TTA Validation Results ({tta_config} configuration)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Baseline Val mIoU: {baseline_stats['Overall mIoU']:.4f}\n")
        f.write(f"TTA Val mIoU: {tta_stats['Overall mIoU']:.4f}\n")
        f.write(f"Improvement: {tta_stats['Overall mIoU'] - baseline_stats['Overall mIoU']:.4f}\n\n")
        f.write("Per-class mIoU:\n")
        f.write("-"*70 + "\n")

        class_names = [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
            'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
            'truck', 'bus', 'train', 'motorcycle', 'bicycle'
        ]

        for i in range(len(class_names)):
            f.write(f"{class_names[i]:<20}: {tta_stats['Class IoU'][i]:.4f}\n")

    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
