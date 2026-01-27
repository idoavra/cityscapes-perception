import segmentation_models_pytorch as smp

def get_model(model_name, encoder_name="efficientnet-b3", classes=19, weights="imagenet"):
    """
    Factory function to fetch different architectures easily.
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
            dropout=0.1,
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

    return model

def count_parameters(model):
    """Utility to see how heavy the model is for your 6GB VRAM"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)