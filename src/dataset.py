import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import v2
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode
from torchvision import tv_tensors
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Cityscapes(Dataset):
    def __init__(self, data_dir, mask_dir, dataset, cache=False, resize=False, transform=None):
        self.transform = transform
        self.data_dir = os.path.join(data_dir, dataset)
        self.mask_dir = os.path.join(mask_dir, dataset)
        self.cache = cache
        self.resize = resize
        
        self.samples = self._make_samples()
        self.cached_samples = []

        if self.cache:
            print(f"Caching {dataset} dataset into RAM...")
            for i in range(len(self.samples)):
                self.cached_samples.append(self._load_sample(i))
            print(f"Done caching {len(self.samples)} samples.")

    def _make_samples(self):
        samples_paths = []
        for city in os.listdir(self.data_dir):
            img_city_dir = os.path.join(self.data_dir, city)
            msk_city_dir = os.path.join(self.mask_dir, city)
            
            for image_name in os.listdir(img_city_dir):
                if not image_name.endswith("_leftImg8bit.png"):
                    continue
                
                image_path = os.path.join(img_city_dir, image_name)
                image_id = image_name.replace("_leftImg8bit.png", "")
                mask_path = os.path.join(msk_city_dir, image_id + "_gtFine_labelTrainIds.png")

                if os.path.exists(mask_path):
                    samples_paths.append((image_path, mask_path))
        return samples_paths

    def _load_sample(self, index):
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        
        if self.resize:
            w, h = image.size
            image = TF.resize(image, size=(h // 2, w // 2), interpolation=InterpolationMode.BILINEAR)
            mask = TF.resize(mask, size=(h // 2, w // 2), interpolation=InterpolationMode.NEAREST)
        
        # Convert mask to tensor early for caching efficiency
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image, mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # 1. Fetch data from cache or disk
        image, mask = self.cached_samples[index] if self.cache else self._load_sample(index)
        
        # 2. Ensure they are NumPy arrays (Albumentations requirement)
        # If they are already NumPy from your _load_sample, you can skip this.
        image = np.array(image)
        mask = np.array(mask)

        if self.transform is not None:
            # 3. Albumentations uses keyword arguments
            augmented = self.transform(image=image, mask=mask)
            
            # 4. Extract the results (ToTensorV2 already converted these to tensors)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Fallback if no transform is provided: manually convert to tensors
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).long()

        # 5. Return processed image and mask
        # .long() is required for the loss function (CrossEntropy/Dice)
        return image, mask.squeeze().long()

def get_cityscapes_loaders(data_dir, mask_dir, batch_size, num_workers=0, resize=True, cache=False, drop_last=False):
    """
    Helper function to create Train, Val, and Test loaders in one go.
    """
    train_transform = A.Compose([
    # 1. Take a 512x512 piece of the 2048x1024 original
    A.RandomCrop(width=512, height=512),

    # 2. Augmentations to reduce overfitting
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
    # A.GaussNoise(p=0.1),  # Uncomment if train/val gap persists after other augmentations

    # 3. Standardize the data
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.CenterCrop(width=512, height=512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    train_set = Cityscapes(data_dir, mask_dir, "train", cache, resize, train_transform)
    full_val_set = Cityscapes(data_dir, mask_dir, "val", cache, resize, val_transform)

    # Split Val into Val and Test (80/20)
    test_size = int(len(full_val_set) * 0.2)
    val_size = len(full_val_set) - test_size
    val_set, test_set = random_split(full_val_set, [val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader