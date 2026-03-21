import matplotlib
matplotlib.use('Agg')

import torch
from tqdm import tqdm
import os

import config
from src.dataset import get_multitask_loaders
from src.multitask_model import MultiTaskModel
from src.losses import Det_Seg_Loss
from src.metrics import StreamSegMetrics
from torch.amp import autocast, GradScaler
from src.utils import save_checkpoint, plot_training

def set_parameter_requires_grad(model, component_name, requires_grad=True):
    """
    Freeze/unfreeze specific model components.

    component_name: 'encoder', 'decoder', 'seghead', 'fpn', 'dethead'
    """
    component_map = {
        'encoder': model.seg_model.encoder,
        'decoder': model.seg_model.decoder,
        'seghead': model.seg_model.segmentation_head,
        'fpn': model.fpn,
        'dethead': model.dethead
    }

    if component_name not in component_map:
        raise ValueError(f"Unknown component: {component_name}")

    for param in component_map[component_name].parameters():
        param.requires_grad = requires_grad

def setup_optimizer_phase1(model):
    """
    Phase 1: Train only FPN + YOLO (freeze backbone + segmentation branch)

    WHY: Don't destroy your 66.6% mIoU segmentation. Warm up detection branch first.
    """
    params = list(model.fpn.parameters()) + list(model.dethead.parameters())
    return torch.optim.Adam(params, lr=config.PHASE1_LR, weight_decay=1e-4)

def setup_optimizer_phase2(model):
    """
    Phase 2: Fine-tune decoder + detection (backbone already frozen by caller).

    Decoder/SegHead get tiny LR to maintain mIoU. FPN/DetHead keep full LR.
    """
    param_groups = [
        {'params': model.seg_model.decoder.parameters(), 'lr': config.PHASE2_DECODER_LR, 'name': 'decoder'},
        {'params': model.seg_model.segmentation_head.parameters(), 'lr': config.PHASE2_DECODER_LR, 'name': 'seghead'},
        {'params': model.fpn.parameters(), 'lr': config.PHASE2_DETECTION_LR, 'name': 'fpn'},
        {'params': model.dethead.parameters(), 'lr': config.PHASE2_DETECTION_LR, 'name': 'dethead'},
    ]
    return torch.optim.Adam(param_groups, weight_decay=1e-4)

def  load_pretrained_segmentation(model, checkpoint_path):
    """
    Load your trained segmentation weights (66.6% mIoU) into the multi-task model.

    Loads: seg_model (encoder + decoder + segmentation_head)
    Skips: fpn, dethead (these are new)
    """
    if not os.path.exists(checkpoint_path):
        print(f"WARNING: Segmentation checkpoint not found at {checkpoint_path}")
        print("Starting with ImageNet encoder only (no pretrained segmentation)")
        return

    print(f"Loading pretrained segmentation from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    seg_state = checkpoint['state_dict']

    # Map old segmentation model keys to new multi-task model keys
    # Old: encoder.xxx → New: seg_model.encoder.xxx
    # Old: decoder.xxx → New: seg_model.decoder.xxx
    # Old: segmentation_head.xxx → New: seg_model.segmentation_head.xxx

    model_dict = model.state_dict()
    pretrained_dict = {}

    for k, v in seg_state.items():
        if k.startswith('encoder.'):
            new_key = 'seg_model.' + k
            if new_key in model_dict:
                pretrained_dict[new_key] = v
        elif k.startswith('decoder.'):
            new_key = 'seg_model.' + k
            if new_key in model_dict:
                pretrained_dict[new_key] = v
        elif k.startswith('segmentation_head.'):
            # Handle index mismatch: checkpoint has .1., model has .0.
            # Checkpoint: segmentation_head.1.weight → Model: seg_model.segmentation_head.0.weight
            new_key = 'seg_model.' + k.replace('segmentation_head.1.', 'segmentation_head.0.')
            if new_key in model_dict:
                pretrained_dict[new_key] = v

    # Load matched weights
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    print(f"Loaded {len(pretrained_dict)} layers from segmentation checkpoint")

from compute_map import compute_detection_map


def main():
    # ==================== TRAINING CONFIGURATION ====================
    # All hyperparameters now in config.py

    # Ensure checkpoint directory exists
    os.makedirs(os.path.dirname(config.MULTITASK_CHECKPOINT), exist_ok=True)

    # ==================== DATA LOADING ====================
    print("Loading multi-task dataset...")
    train_loader, val_loader, _ = get_multitask_loaders(
        data_dir=config.DATA_DIR,
        mask_dir=config.MASK_DIR,
        bbox_dir=config.BBOX_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        resize=False,
        cache=config.CACHE,
        drop_last=config.DROP_LAST
    )

    # ==================== MODEL SETUP ====================
    print("Initializing multi-task model...")
    model = MultiTaskModel(
        num_seg_classes=19,
        num_det_classes=8,
        pretrained='imagenet',
        Bi = config.BiFPN
    ).to(config.DEVICE)

    # Load your trained segmentation weights
    load_pretrained_segmentation(model, config.SEG_CHECKPOINT)

    # ==================== LOSS & METRICS ====================
    criterion = Det_Seg_Loss(
        seg_num_classes=19,
        det_num_classes=8,
        lambda_seg=config.LAMBDA_SEG,
        lambda_det=config.LAMBDA_DET
    )

    seg_metrics = StreamSegMetrics(19)
    scaler = GradScaler()

    # ==================== TRAINING STATE ====================
    best_miou = 0.0
    best_map = 0.0
    start_epoch = 0
    current_phase = 1

    history = {
        "train_loss": [], "train_seg_loss": [], "train_det_loss": [],
        "train_miou": [], "train_map": [],
        "val_loss": [], "val_seg_loss": [], "val_det_loss": [],
        "val_miou": [], "val_map": []
    }

    checkpoint_path = config.MULTITASK_CHECKPOINT

    # ==================== RESUME LOGIC ====================
    if os.path.exists(checkpoint_path) and config.RESUME:
        print(f"Resuming from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)

        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_miou = checkpoint.get('miou', 0.0)
        best_map = checkpoint.get('map', 0.0)
        current_phase = checkpoint.get('phase', 1)
        history = checkpoint.get('history', history)

        print(f"Resumed at Epoch {start_epoch}, Phase {current_phase}")
        print(f"Best mIoU: {best_miou:.4f}, Best mAP: {best_map:.4f}")

    # ==================== PHASE 1: WARM UP DETECTION ====================
    if current_phase == 1:
        print("\n" + "="*60)
        print("PHASE 1: Freeze Backbone + Segmentation, Train FPN + YOLO")
        print("="*60)

        set_parameter_requires_grad(model, 'encoder', requires_grad=False)
        set_parameter_requires_grad(model, 'decoder', requires_grad=False)
        set_parameter_requires_grad(model, 'seghead', requires_grad=False)
        set_parameter_requires_grad(model, 'fpn', requires_grad=True)
        set_parameter_requires_grad(model, 'dethead', requires_grad=True)

        optimizer = setup_optimizer_phase1(model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.PHASE1_EPOCHS, eta_min=1e-6
        )

        phase1_end = min(start_epoch + config.PHASE1_EPOCHS, config.TOTAL_EPOCHS_MULTITASK)
    else:
        phase1_end = config.PHASE1_EPOCHS

    # ==================== TRAINING LOOP ====================
    for epoch in range(start_epoch, config.TOTAL_EPOCHS_MULTITASK):

        # ==================== PHASE TRANSITION ====================
        if epoch == config.PHASE1_EPOCHS and current_phase == 1:
            print("\n" + "="*60)
            print("PHASE 2: Unfreeze All with Differential Learning Rates")
            print("="*60)
            current_phase = 2

            # Unfreeze decoder + detection (encoder stays frozen)
            for component in ['decoder', 'seghead', 'fpn', 'dethead']:
                set_parameter_requires_grad(model, component, requires_grad=True)

            # New optimizer with differential LRs
            optimizer = setup_optimizer_phase2(model)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.PHASE2_EPOCHS, eta_min=1e-7
            )

        # ==================== TRAINING PHASE ====================
        model.train()
        seg_metrics.reset()

        train_loss = 0.0
        train_seg_loss = 0.0
        train_det_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.TOTAL_EPOCHS_MULTITASK} [Train] Phase {current_phase}")

        for i, (images, seg_masks, det_boxes, det_labels) in enumerate(pbar):
            images = images.to(config.DEVICE)
            seg_masks = seg_masks.to(config.DEVICE)

            # Convert detection targets to device
            det_boxes = [boxes.to(config.DEVICE) for boxes in det_boxes]
            det_labels = [labels.to(config.DEVICE) for labels in det_labels]

            with autocast(device_type='cuda'):
                seg_preds, det_preds = model(images)

                # First forward pass: initialize YOLO loss
                loss, loss_dict = criterion(
                    seg_preds, det_preds,
                    seg_masks, det_boxes, det_labels,
                    model=model  # Pass model to initialize YOLO loss
                )
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item()
            train_seg_loss += loss_dict['seg_loss']
            train_det_loss += loss_dict['det_loss']

            # Segmentation metrics
            seg_preds_cls = torch.argmax(seg_preds, dim=1)
            seg_metrics.update(seg_masks, seg_preds_cls)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "seg": f"{loss_dict['seg_loss']:.3f}",
                "det": f"{loss_dict['det_loss']:.3f}"
            })

        train_stats = seg_metrics.get_results()

        # ==================== VALIDATION PHASE ====================
        model.eval()
        seg_metrics.reset()
        val_loss = 0.0
        val_seg_loss = 0.0
        val_det_loss = 0.0

        with torch.no_grad():
            for images, seg_masks, det_boxes, det_labels in val_loader:
                images = images.to(config.DEVICE)
                seg_masks = seg_masks.to(config.DEVICE)
                det_boxes = [boxes.to(config.DEVICE) for boxes in det_boxes]
                det_labels = [labels.to(config.DEVICE) for labels in det_labels]

                with autocast(device_type='cuda'):
                    seg_preds, det_preds = model(images)
                    loss, loss_dict = criterion(
                        seg_preds, det_preds,
                        seg_masks, det_boxes, det_labels,
                        model=model
                    )

                val_loss += loss.item()
                val_seg_loss += loss_dict['seg_loss']
                val_det_loss += loss_dict['det_loss']

                seg_preds_cls = torch.argmax(seg_preds, dim=1)
                seg_metrics.update(seg_masks, seg_preds_cls)

        val_stats = seg_metrics.get_results()
        current_miou = val_stats["Overall mIoU"]

        # Compute mAP
        if (epoch + 1) % 5 == 0 or epoch == config.TOTAL_EPOCHS_MULTITASK - 1:
            current_map = compute_detection_map(model, val_loader, config.DEVICE)
        else:
            current_map = 0.0  # Skip for faster training

        # ==================== LOGGING ====================
        avg_train_loss = train_loss / len(train_loader)
        avg_train_seg = train_seg_loss / len(train_loader)
        avg_train_det = train_det_loss / len(train_loader)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_seg = val_seg_loss / len(val_loader)
        avg_val_det = val_det_loss / len(val_loader)

        history["train_loss"].append(avg_train_loss)
        history["train_seg_loss"].append(avg_train_seg)
        history["train_det_loss"].append(avg_train_det)
        history["train_miou"].append(train_stats['Overall mIoU'])
        history["train_map"].append(0.0)

        history["val_loss"].append(avg_val_loss)
        history["val_seg_loss"].append(avg_val_seg)
        history["val_det_loss"].append(avg_val_det)
        history["val_miou"].append(current_miou)
        history["val_map"].append(current_map)

        print(f"\nEpoch {epoch+1} | Phase {current_phase}")
        print(f"Train - Total: {avg_train_loss:.4f} | Seg: {avg_train_seg:.4f} | Det: {avg_train_det:.4f}")
        print(f"Val   - Total: {avg_val_loss:.4f} | Seg: {avg_val_seg:.4f} | Det: {avg_val_det:.4f}")
        print(f"Metrics - mIoU: {current_miou:.4f} | mAP: {current_map:.4f}")

        # ==================== CHECKPOINTING ====================
        scheduler.step()

        # Save if either metric improves
        if current_miou > best_miou or current_map > best_map:
            if current_miou > best_miou:
                best_miou = current_miou
                print(f"✓ New best mIoU: {best_miou:.4f}")
            if current_map > best_map:
                best_map = current_map
                print(f"✓ New best mAP: {best_map:.4f}")

            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'miou': best_miou,
                'map': best_map,
                'phase': current_phase,
                'history': history
            }, filename=checkpoint_path)

            plot_training(
                history["train_miou"],
                history["train_loss"],
                history["val_miou"],
                history["val_loss"],
                save_dir="plots/multitask"
            )

if __name__ == "__main__":
    main()
