"""
mAP@0.5 computation for the multi-task model (Cityscapes detection).

The model's YOLO head was trained on Cityscapes class IDs (0-7) directly via
v8DetectionLoss, so predictions already use Cityscapes IDs — no COCO remapping.
"""
import torch
import numpy as np
from torchvision.ops import box_iou
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.nms import non_max_suppression


CITYSCAPES_DET_CLASSES = [
    'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle'
]


def compute_detection_map(model, val_loader, device, num_classes=8, iou_thres=0.5, conf_thres=0.001, max_batches=50, max_det=300):
    """
    Compute mAP@0.5 for the multi-task model.

    The YOLO head sits on top of a shared EfficientNet-B3 backbone and was trained
    with Cityscapes detection labels (class IDs 0-7). The 80-channel COCO head is
    reused, but only channels 0-7 received gradients — so predictions already carry
    Cityscapes class IDs. No COCO-to-Cityscapes remapping is needed.
    """
    print(f"\n[mAP] Computing mAP@0.5 (conf_thres={conf_thres}, iou_thres={iou_thres})")

    model.eval()

    # Model outputs Cityscapes class IDs 0-7 directly — keep only those.
    valid_class_ids = list(range(num_classes))

    all_predictions = []
    all_ground_truths = []
    img_id = 0

    with torch.no_grad():
        for batch_idx, (images, seg_masks, det_boxes, det_labels) in enumerate(val_loader):
            if batch_idx >= max_batches:
                break

            images = images.to(device)
            batch_size = images.size(0)

            # Forward pass — returns (seg_logits, det_preds)
            _, det_preds = model(images)

            # Detect head in eval mode returns (decoded, raw_features) tuple.
            if isinstance(det_preds, (list, tuple)):
                det_preds = det_preds[0]

            # Apply NMS — det_preds is (B, 4+nc, num_anchors), matching NMS expectation
            try:
                predictions = non_max_suppression(
                    det_preds,
                    conf_thres=conf_thres,
                    iou_thres=0.45,
                    classes=valid_class_ids,  # Keep only Cityscapes classes [0-7]
                    max_det=max_det
                )

            except Exception as e:
                print(f"[mAP] NMS failed: {e}")
                return 0.0

            # Collect predictions and ground truths
            for i in range(batch_size):
                pred = predictions[i]  # (N, 6) [x1, y1, x2, y2, conf, cls]

                if pred is not None and len(pred) > 0:
                    # No remapping — model already outputs Cityscapes class IDs.
                    for box, conf, cls in zip(pred[:, :4], pred[:, 4], pred[:, 5]):
                        all_predictions.append({
                            'image_id': img_id,
                            'class_id': int(cls.item()),
                            'confidence': float(conf.item()),
                            'bbox': box.cpu().numpy()
                        })

                # Ground truth: convert normalized xywh → pixel xyxy
                if len(det_boxes[i]) > 0:
                    gt_boxes = xywh2xyxy(det_boxes[i]) * 640
                    gt_labels = det_labels[i]

                    for box, label in zip(gt_boxes, gt_labels):
                        all_ground_truths.append({
                            'image_id': img_id,
                            'class_id': int(label.item()),
                            'bbox': box.cpu().numpy()
                        })

                img_id += 1

    # ==================== Compute mAP ====================
    if len(all_predictions) == 0:
        print(f"[mAP] No predictions passed NMS (conf_thres={conf_thres})")
        return 0.0

    if len(all_ground_truths) == 0:
        print(f"[mAP] No ground truths found")
        return 0.0

    print(f"[mAP] Processing: {len(all_predictions)} predictions, {len(all_ground_truths)} ground truths")

    # Per-class AP
    aps = []
    all_ious = []

    for class_id in range(num_classes):
        class_preds = [p for p in all_predictions if p['class_id'] == class_id]
        class_gts = [g for g in all_ground_truths if g['class_id'] == class_id]

        if len(class_gts) == 0:
            continue

        if len(class_preds) == 0:
            aps.append(0.0)
            print(f"  {CITYSCAPES_DET_CLASSES[class_id]:12s}: AP=0.0000 (0 preds, {len(class_gts)} GTs)")
            continue

        # Sort by confidence (descending)
        class_preds = sorted(class_preds, key=lambda x: x['confidence'], reverse=True)

        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        gt_matched = set()

        for pred_idx, pred in enumerate(class_preds):
            pred_img_id = pred['image_id']
            pred_box = torch.tensor(pred['bbox'])

            img_gts = [(idx, g) for idx, g in enumerate(class_gts) if g['image_id'] == pred_img_id]

            if len(img_gts) == 0:
                fp[pred_idx] = 1
                continue

            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in img_gts:
                gt_box = torch.tensor(gt['bbox'])
                iou = box_iou(pred_box.unsqueeze(0), gt_box.unsqueeze(0)).item()

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            all_ious.append(best_iou)

            gt_key = (pred_img_id, best_gt_idx)
            if best_iou >= iou_thres and gt_key not in gt_matched:
                tp[pred_idx] = 1
                gt_matched.add(gt_key)
            else:
                fp[pred_idx] = 1

        # Precision-Recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / len(class_gts)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        # 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.any(recalls >= t):
                ap += np.max(precisions[recalls >= t]) / 11

        aps.append(ap)
        print(f"  {CITYSCAPES_DET_CLASSES[class_id]:12s}: AP={ap:.4f} ({int(tp.sum())}/{len(class_gts)} TP, {len(class_preds)} preds)")

    # IoU statistics
    if len(all_ious) > 0:
        all_ious = np.array(all_ious)
        print(f"[mAP] IoU Statistics:")
        print(f"  Mean IoU: {all_ious.mean():.3f}")
        print(f"  Max IoU:  {all_ious.max():.3f}")
        print(f"  IoU > 0.3: {(all_ious > 0.3).sum()} / {len(all_ious)} ({100*(all_ious > 0.3).mean():.1f}%)")
        print(f"  IoU > 0.5: {(all_ious > 0.5).sum()} / {len(all_ious)} ({100*(all_ious > 0.5).mean():.1f}%)")

    map_score = np.mean(aps) if len(aps) > 0 else 0.0
    print(f"[mAP] mAP@0.5 = {map_score:.4f} (from {len(aps)} classes)\n")
    return map_score
