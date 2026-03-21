"""
Multi-task inference: Segmentation + Detection

Usage:
    # Single image
    python inference.py --image path/to/image.jpg

    # Directory of images (saves results alongside originals)
    python inference.py --dir path/to/images/

    # Quantitative evaluation on test set
    python inference.py --test

    # Custom checkpoint
    python inference.py --image img.jpg --checkpoint checkpoints/Multitask/multitask_best.pth

ROS2 integration:
    from inference import MultiTaskPredictor
    predictor = MultiTaskPredictor("checkpoints/Multitask/multitask_best.pth")
    seg_mask, boxes_xyxy, labels, scores = predictor.predict(image_bgr)
    # seg_mask : (H, W) np.uint8  — Cityscapes class IDs 0-18
    # boxes_xyxy: (N, 4) np.float32 — pixel coords in original image space
    # labels   : (N,)   np.int32   — class IDs 0-7
    # scores   : (N,)   np.float32 — confidence [0, 1]
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.nms import non_max_suppression

import config
from src.multitask_model import MultiTaskModel

# --------------------------------------------------------------------------- #
# Cityscapes class metadata
# --------------------------------------------------------------------------- #

# 19 segmentation train-ID colors (RGB)
SEG_COLORS = np.array([
    [128,  64, 128],  #  0: road
    [244,  35, 232],  #  1: sidewalk
    [ 70,  70,  70],  #  2: building
    [102, 102, 156],  #  3: wall
    [190, 153, 153],  #  4: fence
    [153, 153, 153],  #  5: pole
    [250, 170,  30],  #  6: traffic light
    [220, 220,   0],  #  7: traffic sign
    [107, 142,  35],  #  8: vegetation
    [152, 251, 152],  #  9: terrain
    [ 70, 130, 180],  # 10: sky
    [220,  20,  60],  # 11: person
    [255,   0,   0],  # 12: rider
    [  0,   0, 142],  # 13: car
    [  0,   0,  70],  # 14: truck
    [  0,  60, 100],  # 15: bus
    [  0,  80, 100],  # 16: train
    [  0,   0, 230],  # 17: motorcycle
    [119,  11,  32],  # 18: bicycle
], dtype=np.uint8)

SEG_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

# 8 detection classes
DET_NAMES = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

# Distinct BGR colors for detection boxes (OpenCV uses BGR)
DET_COLORS_BGR = [
    (60,  20, 220),   # person      — red
    (0,  128, 255),   # rider       — orange
    (255, 200,   0),  # car         — cyan-ish
    (255, 100,   0),  # truck       — blue
    (255,   0,   0),  # bus         — dark blue
    (255,   0, 180),  # train       — purple
    (180,   0, 255),  # motorcycle  — magenta
    (0,  220,  90),   # bicycle     — green
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

INPUT_SIZE = 640  # model input resolution


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #

class MultiTaskPredictor:
    """
    Wraps the trained multi-task model for inference.

    Designed to be reused as-is inside a ROS2 node:

        predictor = MultiTaskPredictor(checkpoint_path, device="cuda")
        seg_mask, boxes, labels, scores = predictor.predict(image_bgr)
    """

    def __init__(self, checkpoint_path: str, device: str = None, conf_thres: float = 0.25):
        self.device = torch.device(device or config.DEVICE)
        self.conf_thres = conf_thres

        self.model = MultiTaskModel(num_seg_classes=19, num_det_classes=8).to(self.device)
        self._load_checkpoint(checkpoint_path)
        self.model.eval()

    def _load_checkpoint(self, path: str):
        if not os.path.exists(path):
            sys.exit(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device)
        state = ckpt.get('state_dict', ckpt)
        self.model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {path}")

    def _preprocess(self, image_bgr: np.ndarray):
        """
        BGR uint8 → (1, 3, 640, 640) float32 CUDA tensor.
        Returns scale factors (scale_x, scale_y) so boxes can be mapped back
        to original image space.
        """
        h, w = image_bgr.shape[:2]
        scale_x = w / INPUT_SIZE
        scale_y = h / INPUT_SIZE

        # Resize (not crop) so the full image content is preserved
        resized = cv2.resize(image_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return tensor, scale_x, scale_y, resized

    def predict(self, image_bgr: np.ndarray):
        """
        Run full inference on a BGR image (uint8, any resolution).

        Returns:
            seg_mask  : (H, W) np.uint8  — Cityscapes train IDs 0-18 in ORIGINAL image space
            boxes_xyxy: (N, 4) np.float32 — [x1,y1,x2,y2] pixel coords in ORIGINAL image space
            labels    : (N,)   np.int32   — class IDs 0-7
            scores    : (N,)   np.float32 — confidence scores
        """
        tensor, scale_x, scale_y, _ = self._preprocess(image_bgr)

        with torch.no_grad():
            seg_logits, det_preds = self.model(tensor)

        # --- Segmentation ---
        # Upsample logits to original image size, take argmax
        orig_h, orig_w = image_bgr.shape[:2]
        seg_up = F.interpolate(seg_logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        seg_mask = seg_up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # --- Detection ---
        if isinstance(det_preds, (list, tuple)):
            det_preds = det_preds[0]  # (1, 84, 8400) decoded tensor

        nms_out = non_max_suppression(
            det_preds,
            conf_thres=self.conf_thres,
            iou_thres=0.45,
            classes=list(range(8)),
            max_det=300
        )

        pred = nms_out[0]  # (N, 6): [x1, y1, x2, y2, conf, cls]

        if pred is not None and len(pred) > 0:
            boxes  = pred[:, :4].cpu().numpy().astype(np.float32)
            scores = pred[:,  4].cpu().numpy().astype(np.float32)
            labels = pred[:,  5].cpu().numpy().astype(np.int32)

            # Scale boxes from 640×640 input space → original image space
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        else:
            boxes  = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,),   dtype=np.float32)
            labels = np.zeros((0,),   dtype=np.int32)

        return seg_mask, boxes, labels, scores


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #

def colorize_seg_mask(seg_mask: np.ndarray) -> np.ndarray:
    """(H, W) class IDs → (H, W, 3) BGR color image."""
    color_mask = SEG_COLORS[np.clip(seg_mask, 0, 18)]  # RGB
    return cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)


def draw_detections(image_bgr: np.ndarray, boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Draw bounding boxes + class labels on a BGR image. Returns a copy."""
    out = image_bgr.copy()
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        color = DET_COLORS_BGR[label % len(DET_COLORS_BGR)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{DET_NAMES[label]} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(out, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def visualize(image_bgr: np.ndarray, seg_mask: np.ndarray,
              boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray,
              seg_alpha: float = 0.45) -> np.ndarray:
    """
    Blend segmentation mask + detection boxes onto the original image.

    Returns: BGR image same size as input, ready for cv2.imshow / cv2.imwrite.
    """
    # Resize color mask to match original image if needed
    color_mask = colorize_seg_mask(seg_mask)
    if color_mask.shape[:2] != image_bgr.shape[:2]:
        color_mask = cv2.resize(color_mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    blended = cv2.addWeighted(image_bgr, 1 - seg_alpha, color_mask, seg_alpha, 0)
    result = draw_detections(blended, boxes, labels, scores)
    return result


# --------------------------------------------------------------------------- #
# Quantitative evaluation (test set)
# --------------------------------------------------------------------------- #

def run_test_evaluation(checkpoint_path: str):
    """Compute mAP@0.5 + mIoU on the held-out test split."""
    from src.dataset import get_multitask_loaders
    from src.metrics import StreamSegMetrics
    from compute_map import compute_detection_map

    print("Loading test set...")
    _, _, test_loader = get_multitask_loaders(
        data_dir=config.DATA_DIR,
        mask_dir=config.MASK_DIR,
        bbox_dir=config.BBOX_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        resize=False,
        cache=False,
        drop_last=False
    )

    predictor = MultiTaskPredictor(checkpoint_path, conf_thres=0.001)
    model = predictor.model
    device = predictor.device

    # mAP
    map_score = compute_detection_map(model, test_loader, device, max_batches=9999)

    # mIoU
    seg_metrics = StreamSegMetrics(19)
    model.eval()
    with torch.no_grad():
        for images, seg_masks, _, _ in test_loader:
            images    = images.to(device)
            seg_logits, _ = model(images)
            preds = seg_logits.argmax(dim=1).cpu()
            seg_metrics.update(seg_masks, preds)

    results = seg_metrics.get_results()
    print(f"\nTest set results:")
    print(f"  mIoU  : {results['Overall mIoU']:.4f}")
    print(f"  mAP@0.5: {map_score:.4f}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def process_image(predictor: MultiTaskPredictor, image_path: str, save: bool = True):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Could not read: {image_path}")
        return

    seg_mask, boxes, labels, scores = predictor.predict(image_bgr)
    overlay = visualize(image_bgr, seg_mask, boxes, labels, scores)

    print(f"{os.path.basename(image_path)}: {len(boxes)} detections")
    for box, lbl, score in zip(boxes, labels, scores):
        print(f"  {DET_NAMES[lbl]:12s}  conf={score:.3f}  box=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]")

    if save:
        out_path = os.path.splitext(image_path)[0] + '_inference.jpg'
        cv2.imwrite(out_path, overlay)
        print(f"  Saved → {out_path}")
    else:
        cv2.imshow("Inference", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Multi-task inference: segmentation + detection")
    parser.add_argument('--image',      type=str, help="Path to a single image")
    parser.add_argument('--dir',        type=str, help="Directory of images (.jpg / .png)")
    parser.add_argument('--test',       action='store_true', help="Quantitative eval on test set")
    parser.add_argument('--checkpoint', type=str, default=config.MULTITASK_CHECKPOINT)
    parser.add_argument('--conf',       type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument('--show',       action='store_true', help="Display result (instead of saving)")
    args = parser.parse_args()

    if args.test:
        run_test_evaluation(args.checkpoint)
        return

    predictor = MultiTaskPredictor(args.checkpoint, conf_thres=args.conf)

    if args.image:
        process_image(predictor, args.image, save=not args.show)

    elif args.dir:
        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [
            os.path.join(args.dir, f)
            for f in sorted(os.listdir(args.dir))
            if os.path.splitext(f)[1].lower() in exts
        ]
        if not images:
            sys.exit(f"No images found in {args.dir}")
        for img_path in images:
            process_image(predictor, img_path, save=not args.show)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
