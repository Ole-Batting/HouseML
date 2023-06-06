from typing import Any, Optional

import pytorch_lightning as pl
import torch as T
import torch.nn as nn
import torch.nn.functional as f

from torch.utils.data import random_split, DataLoader, Dataset
from torchvision.transforms import GaussianBlur

from utils.torch_utils import dilate, erode, strel



class ForemanSet(Dataset):
    def __init__(
        self,
        image_size: int = 128,
        noise_level: float = 0.1,
        num_samples: int = 500,
        rand_seed: int = 0,
    ):
        self.image_size = image_size
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.num_colors = min(num_samples, 5)
        self.true_colors = T.rand((self.num_colors, 3))
        self.false_colors = T.rand((self.num_colors, 3))
        self.obsc_colors = T.rand((self.num_colors, 3))
        self.blurrer = GaussianBlur(kernel_size=(3, 3), sigma=2)

        if rand_seed is not None:
            T.manual_seed(rand_seed)

    def __len__(self):
        return self.num_samples

    def noise_mask(self, ssx, ssy):
        x = T.rand(1, 1, self.image_size // ssy, self.image_size // ssx)
        x = T.nn.functional.interpolate(
            x, (self.image_size, self.image_size), mode='bilinear'
        )
        return x[0, 0]
    
    def binary_noise_mask(self, ssx, ssy):
        x = self.noise_mask(ssx, ssy)
        x = (x < 0.5).int()
        return x
    
    def image_noise(self, ssx = 1, ssy = 1):
        x = T.rand(1, 3, self.image_size // ssy, self.image_size // ssx)
        x = T.nn.functional.interpolate(
            x, (self.image_size, self.image_size), mode='bilinear'
        )
        return x[0]

    def __getitem__(self, index):
        # colors
        true_color = self.true_colors[index % self.num_colors]
        false_color = self.false_colors[index % self.num_colors]
        obsc_color = self.obsc_colors[index % self.num_colors]
        print(true_color, false_color, obsc_color)
        diff = true_color - false_color
        diff[T.abs(diff) > 0.3] = 0
        false_color[diff > 0] *= 0.8
        false_color[diff < 0] = 1 - (1 - false_color[diff < 0]) * 0.8
        true_color[diff < 0] *= 0.8
        true_color[diff > 0] = 1 - (1 - true_color[diff > 0]) * 0.8
        print(true_color, false_color, obsc_color)

        # Generate a random binary label for the image
        label = self.binary_noise_mask(8, 8)
        shade = self.binary_noise_mask(48, 2)
        label *= shade
        label = dilate(label.float(), strel((3,3)))
        label = erode(label.float(), strel((5,5)))
        label = dilate(label.float(), strel((5,5)))

        # Generate a random RGB image
        image = label * true_color.view(-1,1,1)
        image += (1 - label) * false_color.view(-1,1,1)
        image = self.blurrer(image)
        sh45 = shade.permute(1,0)
        obsc_level = 0.3
        image[:, sh45==1] *= (1 - obsc_level)
        image[:, sh45==1] += (sh45 * obsc_color.view(-1,1,1))[:, sh45==1] * obsc_level
        image = self.blurrer(image)
        image[:, shade==1] *= (1 - obsc_level)
        image[:, shade==1] += (shade * obsc_color.view(-1,1,1))[:, shade==1] * obsc_level


        # Add synthetic noise to the image
        noise = self.image_noise()
        noise += self.image_noise(16,16)
        noise *= self.noise_level
        noisy_image = image + noise
        print(noisy_image.max())
        noisy_image = T.clamp(noisy_image, 0, 1)

        # Apply the binary label to each channel of the image
        binary_image = noisy_image

        return binary_image, label.float()

class ForemanModule(pl.LightningDataModule):
    def __init__(
        self, 
        image_size: int = 128,
        noise_level: float = 0.1,
        num_samples: int = 500,
        rand_seed: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
        test_ratio: float = 0.2
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.rand_seed = rand_seed
        self.batch_size = batch_size
        self.num_workers = num_workers

        num_test = int(num_samples * test_ratio)
        self.test_split = [num_samples - 2 * num_test, num_test, num_test]


    def prepare_data(self) -> None:
        self.data = ForemanSet(
            image_size=self.image_size,
            noise_level=self.noise_level,
            num_samples=self.num_samples,
            rand_seed=self.rand_seed,
        )


    def setup(self, stage: Optional[str] = None) -> None:
        train, val, test = random_split(
            self.data,
            self.test_split,
            generator=T.Generator().manual_seed(1337)
        )

        if stage in (None, 'fit'):
            self.train_set = train
            self.val_set = val

        if stage in (None, 'test'):
            self.test_set = test


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

foreman500 = ForemanModule()