"""
Test-Time Augmentation (TTA) for Semantic Segmentation
Applies multiple augmentations during inference and averages predictions.
"""
import torch
import torch.nn.functional as F


class TTAWrapper:
    """
    Wrapper for Test-Time Augmentation on semantic segmentation models.

    Supports:
    - Horizontal flip
    - Multi-scale inference (0.75x, 1.0x, 1.25x)
    - Vertical flip (optional, less useful for street scenes)

    Usage:
        tta_model = TTAWrapper(model, scales=[0.75, 1.0, 1.25], flip_h=True)
        logits = tta_model(images)
    """

    def __init__(
        self,
        model,
        scales=[1.0],
        flip_h=True,
        flip_v=False,
        device='cuda'
    ):
        """
        Args:
            model: PyTorch segmentation model
            scales: List of scale factors for multi-scale inference (e.g., [0.75, 1.0, 1.25])
            flip_h: Whether to apply horizontal flip
            flip_v: Whether to apply vertical flip (not recommended for street scenes)
            device: Device to run inference on
        """
        self.model = model
        self.scales = scales
        self.flip_h = flip_h
        self.flip_v = flip_v
        self.device = device

        # Calculate total number of augmentations
        self.num_augs = len(scales)
        if flip_h:
            self.num_augs *= 2
        if flip_v:
            self.num_augs *= 2

        print(f"TTA enabled with {self.num_augs} augmentations:")
        print(f"  - Scales: {scales}")
        print(f"  - Horizontal flip: {flip_h}")
        print(f"  - Vertical flip: {flip_v}")

    def __call__(self, x):
        """
        Performs TTA inference on input batch.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            logits: Averaged logits [B, num_classes, H, W]
        """
        batch_size, _, orig_h, orig_w = x.shape

        # We'll accumulate logits from all augmentations
        accumulated_logits = None

        for scale in self.scales:
            # Resize input if scale != 1.0
            if scale != 1.0:
                new_h = int(orig_h * scale)
                new_w = int(orig_w * scale)
                x_scaled = F.interpolate(
                    x,
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                x_scaled = x

            # Original
            logits = self._forward_and_resize(x_scaled, (orig_h, orig_w))
            accumulated_logits = self._accumulate(accumulated_logits, logits)

            # Horizontal flip
            if self.flip_h:
                x_flipped = torch.flip(x_scaled, dims=[3])  # Flip width dimension
                logits = self._forward_and_resize(x_flipped, (orig_h, orig_w))
                logits = torch.flip(logits, dims=[3])  # Flip back
                accumulated_logits = self._accumulate(accumulated_logits, logits)

            # Vertical flip (optional, less common for Cityscapes)
            if self.flip_v:
                x_flipped = torch.flip(x_scaled, dims=[2])  # Flip height dimension
                logits = self._forward_and_resize(x_flipped, (orig_h, orig_w))
                logits = torch.flip(logits, dims=[2])  # Flip back
                accumulated_logits = self._accumulate(accumulated_logits, logits)

                # Both flips
                if self.flip_h:
                    x_flipped = torch.flip(x_scaled, dims=[2, 3])
                    logits = self._forward_and_resize(x_flipped, (orig_h, orig_w))
                    logits = torch.flip(logits, dims=[2, 3])
                    accumulated_logits = self._accumulate(accumulated_logits, logits)

        # Average all accumulated logits
        averaged_logits = accumulated_logits / self.num_augs

        return averaged_logits

    def _forward_and_resize(self, x, target_size):
        """
        Runs model forward pass and resizes output to target size.

        Args:
            x: Input tensor
            target_size: (height, width) to resize to

        Returns:
            logits: Model output resized to target_size
        """
        with torch.no_grad():
            logits = self.model(x)

            # Resize logits back to original size if needed
            if logits.shape[2:] != target_size:
                logits = F.interpolate(
                    logits,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )

        return logits

    def _accumulate(self, accumulated, new_logits):
        """
        Accumulates logits from multiple augmentations.

        Args:
            accumulated: Previously accumulated logits (or None)
            new_logits: New logits to add

        Returns:
            Updated accumulated logits
        """
        if accumulated is None:
            return new_logits.clone()
        else:
            return accumulated + new_logits


def create_tta_model(model, tta_config='moderate', device='cuda'):
    """
    Factory function to create TTA wrapper with preset configurations.

    Args:
        model: PyTorch segmentation model
        tta_config: One of ['light', 'moderate', 'aggressive']
        device: Device to run on

    Returns:
        TTAWrapper instance
    """
    configs = {
        'light': {
            'scales': [1.0],
            'flip_h': True,
            'flip_v': False
        },
        'moderate': {
            'scales': [0.75, 1.0, 1.25],
            'flip_h': True,
            'flip_v': False
        },
        'aggressive': {
            'scales': [0.5, 0.75, 1.0, 1.25, 1.5],
            'flip_h': True,
            'flip_v': True
        }
    }

    if tta_config not in configs:
        raise ValueError(f"Unknown TTA config: {tta_config}. Choose from {list(configs.keys())}")

    config = configs[tta_config]
    print(f"\nCreating TTA model with '{tta_config}' configuration...")

    return TTAWrapper(
        model=model,
        scales=config['scales'],
        flip_h=config['flip_h'],
        flip_v=config['flip_v'],
        device=device
    )
