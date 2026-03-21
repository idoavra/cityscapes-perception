import os
import json
import random
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
    A.RandomCrop(width=640, height=640),

    # 2. Spatial augmentations (MODERATE - Exp 2.1b revised)
    A.ShiftScaleRotate(
        shift_limit=0.05,     # Reduced: 5% shifts (was 10%)
        scale_limit=0.1,      # Reduced: 0.9x-1.1x (was 0.8-1.2x)
        rotate_limit=10,      # Reduced: ±10° (was ±15°)
        interpolation=1,      # Bilinear for image
        border_mode=0,        # Constant border
        p=0.3                 # Reduced: 30% (was 50%)
    ),
    A.HorizontalFlip(p=0.5),

    # 3. Photometric augmentations (MODERATE)
    A.RandomBrightnessContrast(p=0.25),  # Slightly increased from baseline 0.2
    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05, p=0.3),  # Reduced magnitude
    # GaussNoise removed - too aggressive

    # 4. Light blur (rare)
    A.OneOf([
        A.MotionBlur(blur_limit=3, p=1.0),      # Reduced from 5
        A.GaussianBlur(blur_limit=3, p=1.0),    # Reduced from 5
    ], p=0.1),  # Reduced from 0.2

    # 5. Standardize the data
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.CenterCrop(width=640, height=640),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    train_set = Cityscapes(data_dir, mask_dir, "train", cache, resize, train_transform)
    full_val_set = Cityscapes(data_dir, mask_dir, "val", cache, resize, val_transform)

    # Split Val into Val and Test (80/20)
    test_size = int(len(full_val_set) * 0.2)
    val_size = len(full_val_set) - test_size
    val_set, test_set = random_split(full_val_set, [val_size, test_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


class MultitaskCityscapes(Dataset):
    """
    Multi-task dataset for Cityscapes: Segmentation + Detection

    Returns: (image, seg_mask, det_boxes, det_labels)

    Detection classes (8 total):
        0: person, 1: rider, 2: car, 3: truck,
        4: bus, 5: train, 6: motorcycle, 7: bicycle
    """

    # Mapping from Cityscapes label names to detection class indices
    DET_CLASS_MAP = {
        'person': 0,
        'rider': 1,
        'car': 2,
        'truck': 3,
        'bus': 4,
        'train': 5,
        'motorcycle': 6,
        'bicycle': 7
    }

    def __init__(self, data_dir, mask_dir, bbox_dir, dataset, cache=False, resize=False, transform=None, mosaic_prob=0.0):
        """
        Args:
            data_dir: Path to leftImg8bit directory
            mask_dir: Path to gtFine directory (segmentation masks)
            bbox_dir: Path to gtBbox directory (detection annotations)
            dataset: 'train', 'val', or 'test'
            cache: Cache data in RAM
            resize: Resize images to half resolution
            transform: Albumentations transform (must include bbox_params)
            mosaic_prob: Probability of applying 4-image mosaic augmentation [0, 1]
        """
        self.transform = transform
        self.mosaic_prob = mosaic_prob
        self.data_dir = os.path.join(data_dir, dataset)
        self.mask_dir = os.path.join(mask_dir, dataset)
        self.bbox_dir = os.path.join(bbox_dir, dataset)
        self.cache = cache
        self.resize = resize

        self.samples = self._make_samples()
        self.cached_samples = []

        if self.cache:
            print(f"Caching multi-task {dataset} dataset into RAM...")
            for i in range(len(self.samples)):
                self.cached_samples.append(self._load_sample(i))
            print(f"Done caching {len(self.samples)} samples.")

        # Mosaic sub-transforms (only used when mosaic_prob > 0)
        _bbox_p = A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3)
        self._tile_crop_tf = A.Compose([A.RandomCrop(width=320, height=320)], bbox_params=_bbox_p)
        self._photo_tf = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.25),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05, p=0.3),
            A.OneOf([A.MotionBlur(blur_limit=3, p=1.0), A.GaussianBlur(blur_limit=3, p=1.0)], p=0.1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=_bbox_p)

    def _make_samples(self):
        """
        Create list of (image_path, mask_path, bbox_path) tuples.
        """
        samples_paths = []
        for city in os.listdir(self.data_dir):
            img_city_dir = os.path.join(self.data_dir, city)
            msk_city_dir = os.path.join(self.mask_dir, city)
            bbox_city_dir = os.path.join(self.bbox_dir, city)

            for image_name in os.listdir(img_city_dir):
                if not image_name.endswith("_leftImg8bit.png"):
                    continue

                image_path = os.path.join(img_city_dir, image_name)
                image_id = image_name.replace("_leftImg8bit.png", "")

                # Segmentation mask path
                mask_path = os.path.join(msk_city_dir, image_id + "_gtFine_labelTrainIds.png")

                # Detection annotation path (JSON format)
                # Cityscapes format: {image_id}_gtFine_polygons.json
                bbox_path = os.path.join(bbox_city_dir, image_id + "_gtFine_polygons.json")

                # Only add if both mask and bbox annotations exist
                if os.path.exists(mask_path) and os.path.exists(bbox_path):
                    samples_paths.append((image_path, mask_path, bbox_path))

        return samples_paths

    def _load_bboxes(self, bbox_path, img_width, img_height):
        """
        Load bounding boxes from Cityscapes JSON annotation.

        Args:
            bbox_path: Path to JSON file
            img_width, img_height: Original image dimensions

        Returns:
            boxes: List of [x_center, y_center, width, height] (normalized 0-1)
            labels: List of class indices (0-7)
        """
        # Cityscapes JSON structure:
        # {
        #     "objects": [
        #         {
        #             "label": "car",  # Object class name
        #             "polygon": [[x1,y1], [x2,y2], ...]  # Polygon points
        #         },
        #         ...
        #     ]
        # }

        boxes = []
        labels = []

        with open(bbox_path, 'r') as f:
            data = json.load(f)
        
        for obj in data['objects']:
            label_name = obj['label']

            if label_name not in self.DET_CLASS_MAP:
                continue

            polygon = obj['polygon']
            x_vals = [x[0] for x in polygon]
            y_vals = [y[1] for y in polygon]

            x_min, x_max = min(x_vals), max(x_vals)
            y_min, y_max = min(y_vals), max(y_vals)

            # Clamp pixel coordinates to image boundaries BEFORE normalization
            x_min = max(0, min(img_width - 1, x_min))
            x_max = max(0, min(img_width - 1, x_max))
            y_min = max(0, min(img_height - 1, y_min))
            y_max = max(0, min(img_height - 1, y_max))

            # Now normalize to [0, 1] - will automatically be in valid range
            x_center = ((x_max + x_min) / 2) / img_width
            y_center = ((y_max + y_min) / 2) / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            # Skip degenerate boxes (too small after clamping)
            if width < 0.01 or height < 0.01:  # Less than 1% of image
                continue

            boxes.append([x_center, y_center, width, height])
            labels.append(self.DET_CLASS_MAP[label_name])

        return boxes, labels

    def _load_sample(self, index):
        """
        Load image, segmentation mask, and detection boxes.
        """
        image_path, mask_path, bbox_path = self.samples[index]

        # Load image and mask (same as Cityscapes class)
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        # Get image dimensions for bbox normalization
        img_width, img_height = image.size

        if self.resize:
            w, h = image.size
            image = TF.resize(image, size=(h // 2, w // 2), interpolation=InterpolationMode.BILINEAR)
            mask = TF.resize(mask, size=(h // 2, w // 2), interpolation=InterpolationMode.NEAREST)
            # Update dimensions after resize
            img_width, img_height = w // 2, h // 2

        # Load detection boxes
        boxes, labels = self._load_bboxes(bbox_path, img_width, img_height)

        # Convert mask to tensor for caching
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        return image, mask, boxes, labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Returns:
            image: (3, H, W) tensor
            mask: (H, W) tensor with class indices
            boxes: (N, 4) tensor [x_center, y_center, w, h] normalized
            labels: (N,) tensor with class indices
        """
        # 0. Mosaic augmentation (bypasses standard transform pipeline)
        if self.mosaic_prob > 0 and random.random() < self.mosaic_prob:
            return self._mosaic(index)

        # 1. Fetch data
        image, mask, boxes, labels = self.cached_samples[index] if self.cache else self._load_sample(index)

        # 2. Convert to NumPy for Albumentations
        image = np.array(image)
        mask = np.array(mask)

        # 3. Apply transforms (if boxes exist)
        if self.transform is not None and len(boxes) > 0:
            # Albumentations with format='yolo' expects YOLO format boxes
            # Our boxes are already in YOLO format (x_center, y_center, w, h), so pass directly
            augmented = self.transform(image=image, mask=mask, bboxes=boxes, class_labels=labels)
            image = augmented['image']
            mask = augmented['mask']
            boxes = augmented['bboxes']
            labels = augmented['class_labels']
        elif self.transform is not None:
            # No boxes in this image - pass empty lists (Albumentations requires them)
            augmented = self.transform(image=image, mask=mask, bboxes=[], class_labels=[])
            image = augmented['image']
            mask = augmented['mask']
        else:
            # No transform - manual conversion
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).long()

        # 4. Convert boxes and labels to tensors
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            # Empty tensors for images with no objects
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        return image, mask.squeeze().long(), boxes, labels

    def _mosaic(self, index):
        """
        4-image mosaic augmentation.

        Stitches 4 random 320x320 crops into a single 640x640 canvas.
        Quadrant layout:
            TL (col=0,   row=0  ) | TR (col=320, row=0  )
            ----------------------+----------------------
            BL (col=0,   row=320) | BR (col=320, row=320)

        Box coordinate transform per tile (tile-normalized → mosaic-normalized):
            mx = bx * 0.5 + x_off   (x_off ∈ {0.0, 0.5})
            my = by * 0.5 + y_off   (y_off ∈ {0.0, 0.5})
            mw = bw * 0.5
            mh = bh * 0.5
        """
        TILE = 320
        SIZE = 640
        # (x_offset, y_offset) in mosaic-normalized space for each quadrant
        offsets = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)]
        indices = [index] + [random.randint(0, len(self) - 1) for _ in range(3)]

        canvas_img  = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        canvas_mask = np.full((SIZE, SIZE), 255, dtype=np.int64)  # 255 = ignore index
        all_boxes   = []
        all_labels  = []

        for i, idx in enumerate(indices):
            img, mask, boxes, labels = (
                self.cached_samples[idx] if self.cache else self._load_sample(idx)
            )
            img  = np.array(img)
            mask = np.array(mask)

            # Crop each tile to 320x320 (boxes clipped by min_visibility=0.3)
            boxes  = list(boxes)   if not isinstance(boxes, list)  else boxes
            labels = list(labels)  if not isinstance(labels, list) else labels
            cropped = self._tile_crop_tf(
                image=img, mask=mask,
                bboxes=boxes, class_labels=labels
            )
            tile_img    = cropped['image']   # (320, 320, 3)
            tile_mask   = cropped['mask']    # (320, 320)
            tile_boxes  = cropped['bboxes']
            tile_labels = cropped['class_labels']

            # Place tile into canvas
            x_off, y_off = offsets[i]
            col = int(x_off * SIZE)
            row = int(y_off * SIZE)
            canvas_img [row:row+TILE, col:col+TILE] = tile_img
            canvas_mask[row:row+TILE, col:col+TILE] = tile_mask

            # Remap boxes from tile space → mosaic space
            for box, lbl in zip(tile_boxes, tile_labels):
                bx, by, bw, bh = box
                all_boxes.append([bx * 0.5 + x_off, by * 0.5 + y_off, bw * 0.5, bh * 0.5])
                all_labels.append(lbl)

        # Apply photometric augmentation + normalize + ToTensor to the full canvas
        result = self._photo_tf(
            image=canvas_img, mask=canvas_mask,
            bboxes=all_boxes, class_labels=all_labels
        )
        image  = result['image']
        mask   = result['mask']
        boxes  = result['bboxes']
        labels = result['class_labels']

        if len(boxes) > 0:
            boxes  = torch.tensor(boxes,  dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.long)

        return image, mask.squeeze().long(), boxes, labels


def multitask_collate_fn(batch):
    """
    Custom collate function for multi-task batches with variable-length boxes.

    Args:
        batch: List of (image, mask, boxes, labels) tuples

    Returns:
        images: (B, 3, H, W) stacked tensor
        masks: (B, H, W) stacked tensor
        boxes: List of (N_i, 4) tensors (variable length per image)
        labels: List of (N_i,) tensors (variable length per image)
    """
    images, masks, boxes, labels = zip(*batch)

    images = torch.stack(images)
    masks = torch.stack(masks)
    # Keep boxes and labels as lists (each image has different number of objects)

    return images, masks, list(boxes), list(labels)


def get_multitask_loaders(data_dir, mask_dir, bbox_dir, batch_size, num_workers=0, resize=True, cache=False, drop_last=False):
    """
    Create multi-task dataloaders for training, validation, and testing.

    Args:
        data_dir: Path to leftImg8bit
        mask_dir: Path to gtFine
        bbox_dir: Path to gtBbox (detection annotations)
        batch_size: Batch size
        num_workers: Number of worker processes
        resize: Resize to half resolution
        cache: Cache data in RAM
        drop_last: Drop incomplete batches

    Returns:
        train_loader, val_loader, test_loader
    """
    train_transform = A.Compose([
        A.RandomCrop(width=640, height=640),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=10,
            interpolation=1,
            border_mode=0,
            p=0.3
        ),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.25),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05, p=0.3),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
        ], p=0.1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3)
    )

    val_transform = A.Compose([
        A.CenterCrop(width=640, height=640),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3)
    )

    train_set = MultitaskCityscapes(data_dir, mask_dir, bbox_dir, "train", cache, resize, train_transform, mosaic_prob=0.5)
    full_val_set = MultitaskCityscapes(data_dir, mask_dir, bbox_dir, "val", cache, resize, val_transform, mosaic_prob=0.0)

    # Split Val into Val and Test (80/20)
    test_size = int(len(full_val_set) * 0.2)
    val_size = len(full_val_set) - test_size
    val_set, test_set = random_split(full_val_set, [val_size, test_size],
                                     generator=torch.Generator().manual_seed(42))

    # Use custom collate function to handle variable-length boxes
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=multitask_collate_fn
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=multitask_collate_fn
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=multitask_collate_fn
    )

    return train_loader, val_loader, test_loader