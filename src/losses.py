import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss

class JointLoss(nn.Module):
    def __init__(self, num_classes=19, alpha=0.5, gamma=2.0, weight=None):
        super(JointLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        # Standard CE Loss
        self.ce = nn.CrossEntropyLoss(ignore_index=255, weight=weight)
        
    def focal_loss(self, logits, target):
        """
        Focal Loss = - (1 - pt)^gamma * log(pt)
        """
        # 1. Get standard Cross Entropy per pixel
        ce_loss = self.ce(logits, target) 
        
        # 2. Calculate probability of the correct class (pt)
        pt = torch.exp(-ce_loss) 
        
        # 3. Calculate Focal weight: (1-pt)^gamma
        # Pixels with high probability (pt -> 1) get a weight near 0
        # Pixels with low probability (pt -> 0) get a high weight
        f_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # 4. Average the loss across the non-ignored pixels
        return f_loss.mean()
    
    def dice_loss(self, logits, target):
        """Vectorized Dice Loss (No Loops)"""
        smooth = 1e-7
        probs = F.softmax(logits, dim=1)
        
        # Create one-hot target: (B, H, W) -> (B, C, H, W)
        # We mask out the 255 values so they don't contribute to the sum
        valid_mask = (target != 255).float().unsqueeze(1)
        target_fixed = target.clone()
        target_fixed[target == 255] = 0
        target_one_hot = F.one_hot(target_fixed, self.num_classes).permute(0, 3, 1, 2).float()
        
        # Multiply by valid mask to ignore the 255 regions
        target_one_hot = target_one_hot * valid_mask
        probs = probs * valid_mask

        intersection = torch.sum(probs * target_one_hot, dim=(0, 2, 3))
        union = torch.sum(probs, dim=(0, 2, 3)) + torch.sum(target_one_hot, dim=(0, 2, 3))
        
        dice_per_class = (2. * intersection + smooth) / (union + smooth)
        return 1.0 - dice_per_class.mean()

    def forward(self, logits, target):
        focal = self.focal_loss(logits, target)
        dice_loss = self.dice_loss(logits, target)
        
        # Total loss = CE + (alpha * Dice)
        return focal + (self.alpha * dice_loss)
    

class Det_Seg_Loss(nn.Module):
    """
    Multi-task loss: L_total = λ_seg × (Focal+Dice) + λ_det × (Box+Cls+DFL)
    """
    def __init__(self, seg_num_classes=19, det_num_classes=8,
                 alpha=0.5, gamma=2.0, weight=None,
                 lambda_seg=1.0, lambda_det=1.0):
        super().__init__()

        self.seg_num_classes = seg_num_classes
        self.det_num_classes = det_num_classes
        self.seg_loss = JointLoss(num_classes=seg_num_classes, alpha=alpha, gamma=gamma, weight=weight)

        # Lazy init: v8DetectionLoss needs model.dethead to extract stride/anchor params
        self.det_loss_fn = None
        self.lambda_seg = lambda_seg
        self.lambda_det = lambda_det

    def _init_det_loss(self, model):
        """Initialize YOLO loss with full YOLO model (needs model.args)"""
        if self.det_loss_fn is None:
            self.det_loss_fn = v8DetectionLoss(model.yolo_model)

    def forward(self, seg_preds, det_preds, seg_targets, det_targets, det_labels, model=None):
        """
        Args:
            seg_preds: (B, 19, H, W)
            det_preds: List[Tensor] from P3, P4, P5
            seg_targets: (B, H, W)
            det_targets: List[(N_i, 4)] YOLO format [x, y, w, h]
            det_labels: List[(N_i,)]
            model: MultiTaskModel
        """
        loss_seg = self.seg_loss(seg_preds, seg_targets)

        # Initialize YOLO loss on first call
        if model is not None and self.det_loss_fn is None:
            self._init_det_loss(model)

        if self.det_loss_fn is not None:
            # Convert list format to YOLO batch format
            batch_bboxes, batch_cls, batch_idx = [], [], []

            for i, (boxes, labels) in enumerate(zip(det_targets, det_labels)):
                if len(boxes) > 0:
                    batch_bboxes.append(boxes)
                    batch_cls.append(labels)
                    batch_idx.append(torch.full((len(boxes),), i, dtype=torch.long, device=boxes.device))

            if len(batch_bboxes) > 0:
                batch_dict = {
                    'cls': torch.cat(batch_cls).unsqueeze(1),
                    'bboxes': torch.cat(batch_bboxes),
                    'batch_idx': torch.cat(batch_idx).unsqueeze(1),
                }

                loss_det, loss_items = self.det_loss_fn(det_preds, batch_dict)

                # YOLO returns [box, cls, dfl] losses - sum for total
                if isinstance(loss_det, torch.Tensor) and loss_det.numel() > 1:
                    loss_det = loss_det.sum()

                # v8DetectionLoss multiplies by batch_size internally for optimizer
                # consistency. Undo it so lambda_det operates on per-sample scale,
                # matching the seg loss scale.
                batch_size = seg_preds.shape[0]
                loss_det = loss_det / batch_size

                det_box_loss = loss_items[0].item() if len(loss_items) > 0 else 0.0
                det_cls_loss = loss_items[1].item() if len(loss_items) > 1 else 0.0
                det_dfl_loss = loss_items[2].item() if len(loss_items) > 2 else 0.0
            else:
                # No boxes in batch - create zero loss with gradient support
                loss_det = torch.tensor(0.0, device=seg_preds.device, requires_grad=True)
                det_box_loss = det_cls_loss = det_dfl_loss = 0.0
        else:
            # Detection loss not initialized yet
            loss_det = torch.tensor(0.0, device=seg_preds.device, requires_grad=True)
            det_box_loss = det_cls_loss = det_dfl_loss = 0.0

        total_loss = self.lambda_seg * loss_seg + self.lambda_det * loss_det

        loss_dict = {
            'seg_loss': loss_seg.item(),
            'det_box_loss': det_box_loss,
            'det_cls_loss': det_cls_loss,
            'det_dfl_loss': det_dfl_loss,
            'det_loss': loss_det.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict
