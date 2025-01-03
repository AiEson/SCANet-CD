import torch
import cv2
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import numpy as np
import albumentations as A
from .utils import obtain_cutmix_box


def get_train_transform(img_size=512):
    trfm = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.75),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(),
                    A.RandomGamma(),
                    A.ColorJitter(
                        brightness=0.47,
                        contrast=0.47,
                        saturation=0.41,
                        hue=0.21,
                        always_apply=False,
                        p=0.8,
                    ),
                ],
                p=0.2,
            ),
            A.OneOf(
                [
                    A.ElasticTransform(
                        alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
                    ),
                    A.GridDistortion(),
                    A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                ],
                p=0.3,
            )
        ]
    )
    return trfm


class WHUChangeDetectionDataset(Dataset):
    def __init__(
        self,
        dataset_root,
        ids_filepath,
        transform=get_train_transform(),
        test_mode=False,
        size=512,
    ):
        self.paths = None
        self.transform = get_train_transform(img_size=size)
        self.test_mode = test_mode
        self.data_root = dataset_root
        self.size = size

        with open(ids_filepath, "r") as f:
            self.paths = [x.strip() for x in f.readlines()]
            
        # Determine split based on ids_filepath
        if 'train' in ids_filepath:
            self.split = 'train'
        elif 'test' in ids_filepath:
            self.split = 'test'
        elif 'val' in ids_filepath:
            self.split = 'val'
        else:
            raise ValueError(f"Could not determine split from ids_filepath: {ids_filepath}")

        self.to_tensor = T.ToTensor()

    def __getitem__(self, index):
        # Load image A and B
        img_a = Image.open(f"{self.data_root}/{self.split}/A/{self.paths[index]}")
        img_b = Image.open(f"{self.data_root}/{self.split}/B/{self.paths[index]}")
        mask = Image.open(f"{self.data_root}/{self.split}/label/{self.paths[index]}")

        # Convert to RGB and grayscale
        img_a = img_a.convert("RGB")
        img_b = img_b.convert("RGB")
        mask = mask.convert("L")
        
        # Convert to numpy arrays
        img_a = np.array(img_a)
        img_b = np.array(img_b)
        mask = np.array(mask) / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        if not self.test_mode:
            # Apply same augmentation to both images
            augments_a = self.transform(image=img_a) if self.transform else {"image": img_a}
            augments_b = self.transform(image=img_b) if self.transform else {"image": img_b}
            
            # Convert to tensors separately
            img_a_tensor = self.to_tensor(augments_a["image"])
            img_b_tensor = self.to_tensor(augments_b["image"])
            
            cutmix_box = obtain_cutmix_box(self.size, p=0.5)
            return (
                img_a_tensor,
                img_b_tensor,
                self.to_tensor(mask),
                cutmix_box.unsqueeze(0),
            )
        else:
            # Convert to tensors separately
            img_a_tensor = self.to_tensor(img_a)
            img_b_tensor = self.to_tensor(img_b)
            return img_a_tensor, img_b_tensor, self.to_tensor(mask)

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return len(self.paths)


def get_mean_std(loader):
    # Var[x] = E[X**2]-E[X]**2
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    print(num_batches)
    print(channels_sum)
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import cv2

    wdataset = WHUChangeDetectionDataset(
        dataset_root="datasets/datasets/WHU",
        ids_filepath="datasets/datasets/WHU/train.txt",
        test_mode=True
    )
