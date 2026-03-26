import numpy as np
import os
from urllib.request import urlretrieve
from typing import Any
from ._base import register_dataset

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from lightning import LightningDataModule

@register_dataset('bmnist')
class BinaryMNIST(Dataset):
    def __init__(self, root, split, indices=None, download=True, flatten=True, k=2, num_cls=10):
        self.base_dataset = datasets.MNIST(root=root, train=(split == 'train' or split == 'valid'), download=download)
        self.indices = indices if indices is not None else list(range(len(self.base_dataset)))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float())
        ])
        self.flatten = flatten
        self.num_cls = num_cls
        self.k = k
        print(f'self.base_dataset.data shape: {self.base_dataset.data.shape}')
        reshaped_data = self.base_dataset.data.view(-1, 28*28).float()
        self.probs = torch.stack([(reshaped_data > 0.5).float(), 1.0 - (reshaped_data > 0.5).float()], dim=1).contiguous().transpose(1, 2) # (N, 2, 784) -> (N, 784, 2)
        print(f'self.probs shape: {self.probs.shape}')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.base_dataset[real_idx]   # 這裡拿到的是 PIL image
        img = self.transform(img) # (1, 28, 28), binary tensor
        img = img.squeeze(0) # (28, 28)
        # img = torch.stack([img, 1 - img], dim=-1)  # (H, W, 2)
        if self.flatten:
            img = img.reshape(img.shape[0] * img.shape[1]).long()  # (H*W, )
        return img, label

class BinaryMNISTDataModule(LightningDataModule):
    """
    Binary MNIST data module.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_split: str = "train",
        val_split: str = "valid",
        test_split: str = "test",
        k=2,
        dim=784,
        with_labels=True,
    ):
        """Initialize a `BinaryMNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        :param train_split: Name of train split file suffix. Defaults to `"train"`.
        :param val_split: Name of validation split file suffix. Defaults to `"valid"`.
        :param test_split: Name of test split file suffix. Defaults to `"test"`.
        :param transform: Optional transform applied to each sample.
        """
        super().__init__()

        # store hyperparameters in ckpt & accessible via self.hparams
        self.save_hyperparameters(logger=False)

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None
        self.batch_size_per_device = batch_size
        self.k = k
        self.dim = dim
        self.with_labels = with_labels

    def prepare_data(self) -> None:
        """Download data if needed.

        Lightning guarantees this is called only from rank 0 in distributed settings.
        """

    def setup(self, stage: str | None = None) -> None:
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            train_indices = list(range(50000))
            self.data_train = BinaryMNIST(
                root=self.hparams.data_dir,
                split=self.hparams.train_split,
                indices=train_indices,
                flatten=True,
                k=self.k,
            )
            
            val_indices = list(range(50000,60000))
            self.data_val = BinaryMNIST(
                root=self.hparams.data_dir,
                split=self.hparams.val_split,
                indices=val_indices,
                flatten=True,
                k=self.k,
            )

            test_indices = list(range(10000))
            self.data_test = BinaryMNIST(
                root=self.hparams.data_dir,
                split=self.hparams.test_split,
                indices=test_indices,
                flatten=True,
                k=self.k,
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        assert self.data_train
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
        assert self.data_val
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader."""
        assert self.data_test
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True,
        )

    def teardown(self, stage: str | None = None) -> None:
        """Lightning hook for cleaning up after fit/validate/test/predict."""

    def state_dict(self) -> dict[Any, Any]:
        """Called when saving a checkpoint."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint."""
        # nothing to restore
        return