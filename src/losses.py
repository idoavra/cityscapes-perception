import torch
import torch.nn as nn
import torch.nn.functional as F

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