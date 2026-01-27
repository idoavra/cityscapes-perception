import torch
import torch.nn as nn
import torch.nn.functional as F

class JointLoss(nn.Module):
    def __init__(self, num_classes=19, alpha=0.5, weight=None):
        super(JointLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        # Standard CE Loss
        self.ce = nn.CrossEntropyLoss(ignore_index=255, weight=weight)

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
        ce_loss = self.ce(logits, target)
        dice_loss = self.dice_loss(logits, target)
        
        # Total loss = CE + (alpha * Dice)
        return ce_loss + (self.alpha * dice_loss)