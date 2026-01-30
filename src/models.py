import segmentation_models_pytorch as smp
import torch.nn as nn

def get_model(model_name, encoder_name="efficientnet-b3", classes=19, weights="imagenet", dropout=0.0):
    """
    Factory function to fetch different architectures easily.

    Args:
        model_name: Architecture type ('unet', 'deeplabv3plus', 'manet')
        encoder_name: Backbone encoder
        classes: Number of segmentation classes
        weights: Pretrained weights ('imagenet' or None)
        dropout: Dropout probability for segmentation head (0.0 = disabled)
    """
    
    if model_name.lower() == "unet":
        # Your original baseline
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=weights,
            in_channels=3,
            classes=classes,
            activation=None # We use CrossEntropy/Dice on raw logits
        )
    
    elif model_name.lower() == "deeplabv3plus":
        # The recommended upgrade for Street Scenes
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=weights,
            in_channels=3,
            classes=classes,
            dropout=dropout,
            activation=None
        )
        
    elif model_name.lower() == "manet":
        # Great alternative: uses attention but is lighter than DeepLab
        model = smp.MAnet(
            encoder_name=encoder_name,
            encoder_weights=weights,
            in_channels=3,
            classes=classes,
        )
        
    else:
        raise ValueError(f"Model {model_name} not recognized. Choose unet, deeplabv3plus, or manet.")

    # Add extra dropout to segmentation head if requested
    if dropout > 0:
        model.segmentation_head = nn.Sequential(
            nn.Dropout2d(p=dropout),
            *list(model.segmentation_head.children())
        )

    return model

def count_parameters(model):
    """Utility to see how heavy the model is for your 6GB VRAM"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)