import os
from enum import Enum

import PIL
import torch
from torchvision import transforms, datasets


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source,
        classname,
        input_size=518,
        output_size=224,
        split=DatasetSplit.TEST,
        external_transform=None,
        external_gt_transform=None,
        **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.classname = classname

        if external_transform is None:
            self.transform_img = [
                transforms.Resize((input_size, input_size)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
            self.transform_img = transforms.Compose(self.transform_img)
        else:
            self.transform_img = external_transform

        # Output size of the mask has to be of shape: 1×224×224
        if external_gt_transform is None:
            self.transform_mask = [
                transforms.Resize((output_size, output_size)),
                transforms.CenterCrop(output_size),
                transforms.ToTensor(),
            ]
            self.transform_mask = transforms.Compose(self.transform_mask)
        else:
            self.transform_mask = external_gt_transform
        self.output_shape = (1, output_size, output_size)
        self.image_paths = []
        self.mask_paths = []
        self.get_image_data()

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask) > 0
        else:
            mask = torch.zeros([*self.output_shape])

        return {
            "image": image,
            "mask": mask,
            "is_anomaly": int("good" not in image_path),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.image_paths)

    def get_image_data(self):
        path = os.path.join(self.source, self.classname, self.split.value)
        images = datasets.ImageFolder(path)
        self.image_paths = []
        self.mask_paths = []
        for item, _ in sorted(images.imgs):
            self.image_paths.append(item)
            file_format = item.split(".")[-1]
            mask_path = item.replace("test", "ground_truth").replace(
                f".{file_format}", f"_mask.{file_format}"
            )
            if (
                (self.split == DatasetSplit.TEST)
                and ("good" not in item)
                and os.path.exists(mask_path)
            ):
                self.mask_paths.append(mask_path)
            else:
                self.mask_paths.append(None)
