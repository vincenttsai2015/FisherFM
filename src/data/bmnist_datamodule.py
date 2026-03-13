import numpy as np
import os
from urllib.request import urlretrieve
from typing import Any
from ._base import register_dataset

import torch
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule

@register_dataset('bmnist')
class BinaryMNIST(Dataset):
    """
    Binarized MNIST dataset.
    """
    data_url = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat"

    def __init__(self, root, split, with_labels=True, transform=None,
                 labels_root='data/bmnist', val_from_train=10000, num_cls=10, k=2, dim=784):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.with_labels = with_labels
        self.num_cls = num_cls
        self.k = k
        self.dim = dim

        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, f"binarized_mnist_{split}.amat")
        if not os.path.exists(path):
            print(f"Downloading {split} set...")
            urlretrieve(self.data_url.format(split), path)

        # data: float32 0/1, shape (N, 784)
        data = np.loadtxt(path).astype(np.float32)
        # self.data = torch.from_numpy(np.loadtxt(path).astype(np.float32))
        # turn into tokens: int64 0/1, shape (N, 784)
        self.seq = torch.from_numpy(data).round().to(torch.long)
        print(f'Loaded {split} set with shape {self.seq.shape}.')

        self.targets = None
        self.probs = torch.stack([self.seq, 1.0 - self.seq], dim=1).contiguous().transpose(1, 2) # (N, 2, 784) -> (N, 784, 2)
        if self.with_labels:
            from torchvision.datasets import MNIST
            labels_root = labels_root or root
            mnist_train = MNIST(labels_root, train=True, download=True)
            mnist_test = MNIST(labels_root, train=False, download=True)

            if split == "train":
                targets = mnist_train.targets[:-val_from_train]
            elif split == "valid":
                targets = mnist_train.targets[-val_from_train:]
            elif split == "test":
                targets = mnist_test.targets
            else:
                raise ValueError(split)

            self.targets = targets.to(torch.long)
            print(f'Loaded {split} labels with shape {self.targets.shape}.')

            if len(self.targets) != self.seq.shape[0]:
                raise RuntimeError(
                    f"Label length mismatch split={split}: data={self.seq.shape[0]} vs labels={len(self.targets)}. "
                    f"Your split alignment assumption may be wrong."
                )

    def __len__(self):
        return self.seq.size(0)

    def __getitem__(self, idx):
        seq = self.seq[idx] 
        # seq = torch.stack([seq, 1 - seq], dim=-1)
        # seq = (seq > 0.5).long()
        # if self.transform is not None: 
        #     seq = self.transform(seq)
        # seq = self.seq[idx].float() # (784,) long in {0,1}
        # seq = seq.view(self.k, self.dim) # (B, dim, k) -> (B, k, dim)
        if self.targets is None:
            # 如果你真的沒 label，先給 dummy label（見路線2），但這只會讓 loss 沒意義
            cls = torch.tensor(0, dtype=torch.long)
            return seq
        else:
            cls = self.targets[idx]    # scalar long 0..9
            return seq, cls

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
            self.data_train = BinaryMNIST(
                root=self.hparams.data_dir,
                split=self.hparams.train_split,
                with_labels=self.with_labels,
            )
            self.data_val = BinaryMNIST(
                root=self.hparams.data_dir,
                split=self.hparams.val_split,
                with_labels=self.with_labels,
            )
            self.data_test = BinaryMNIST(
                root=self.hparams.data_dir,
                split=self.hparams.test_split,
                with_labels=self.with_labels,
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