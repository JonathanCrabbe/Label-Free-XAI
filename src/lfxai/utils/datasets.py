import abc
import logging
import os
import pathlib
import subprocess
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
import wget
from PIL import Image
from scipy.io.arff import loadarff
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST

"""
The code for DSprites is adapted from https://github.com/YannDubs/disentangling-vae/blob/master/utils/datasets.py
"""


DIR = os.path.abspath(os.path.dirname(__file__))
COLOUR_BLACK = 0
COLOUR_WHITE = 1
DATASETS_DICT = {"dsprites": "DSprites"}
DATASETS = list(DATASETS_DICT.keys())


class DisentangledDataset(Dataset, abc.ABC):
    """Base Class for disentangled VAE datasets.

    Parameters:
    ----------
    root : string
        Root directory of dataset.
    transforms_list : list
        List of `torch.vision.transforms` to apply to the data when loading it.
    """

    def __init__(self, root, transforms_list=[], logger=logging.getLogger(__name__)):
        self.root = root
        self.train_data = os.path.join(root, type(self).files["train"])
        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger

        if not os.path.isdir(root):
            self.logger.info(f"Downloading {str(type(self))} ...")
            self.download()
            self.logger.info("Finished Downloading.")

    def __len__(self):
        return len(self.imgs)

    @abc.abstractmethod
    def __getitem__(self, idx):
        """Get the image of `idx`.
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.
        """
        pass

    @abc.abstractmethod
    def download(self):
        """Download the dataset."""
        pass


class DSprites(DisentangledDataset):
    """DSprites Dataset from [1].
    Disentanglement test Sprites dataset.Procedurally generated 2D shapes, from 6
    disentangled latent factors. This dataset uses 6 latents, controlling the color,
    shape, scale, rotation and position of a sprite. All possible variations of
    the latents are present. Ordering along dimension 1 is fixed and can be mapped
    back to the exact latent values that generated that image. Pixel outputs are
    different. No noise added.
    Notes
    -----
    - Link : https://github.com/deepmind/dsprites-dataset/
    - hard coded metadata because issue with python 3 loading of python 2

    Parameters:
    ----------
    root : string
        Root directory of dataset.
    References
    ----------
    [1] Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick,
        M., ... & Lerchner, A. (2017). beta-vae: Learning basic visual concepts
        with a constrained variational framework. In International Conference
        on Learning Representations.
    """

    urls = {
        "train": "https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"
    }
    files = {"train": "dsprite_train.npz"}
    lat_names = ("shape", "scale", "orientation", "posX", "posY")
    lat_sizes = np.array([3, 6, 40, 32, 32])
    img_size = (1, 64, 64)
    background_color = COLOUR_BLACK
    lat_values = {
        "posX": np.array(
            [
                0.0,
                0.03225806,
                0.06451613,
                0.09677419,
                0.12903226,
                0.16129032,
                0.19354839,
                0.22580645,
                0.25806452,
                0.29032258,
                0.32258065,
                0.35483871,
                0.38709677,
                0.41935484,
                0.4516129,
                0.48387097,
                0.51612903,
                0.5483871,
                0.58064516,
                0.61290323,
                0.64516129,
                0.67741935,
                0.70967742,
                0.74193548,
                0.77419355,
                0.80645161,
                0.83870968,
                0.87096774,
                0.90322581,
                0.93548387,
                0.96774194,
                1.0,
            ]
        ),
        "posY": np.array(
            [
                0.0,
                0.03225806,
                0.06451613,
                0.09677419,
                0.12903226,
                0.16129032,
                0.19354839,
                0.22580645,
                0.25806452,
                0.29032258,
                0.32258065,
                0.35483871,
                0.38709677,
                0.41935484,
                0.4516129,
                0.48387097,
                0.51612903,
                0.5483871,
                0.58064516,
                0.61290323,
                0.64516129,
                0.67741935,
                0.70967742,
                0.74193548,
                0.77419355,
                0.80645161,
                0.83870968,
                0.87096774,
                0.90322581,
                0.93548387,
                0.96774194,
                1.0,
            ]
        ),
        "scale": np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        "orientation": np.array(
            [
                0.0,
                0.16110732,
                0.32221463,
                0.48332195,
                0.64442926,
                0.80553658,
                0.96664389,
                1.12775121,
                1.28885852,
                1.44996584,
                1.61107316,
                1.77218047,
                1.93328779,
                2.0943951,
                2.25550242,
                2.41660973,
                2.57771705,
                2.73882436,
                2.89993168,
                3.061039,
                3.22214631,
                3.38325363,
                3.54436094,
                3.70546826,
                3.86657557,
                4.02768289,
                4.1887902,
                4.34989752,
                4.51100484,
                4.67211215,
                4.83321947,
                4.99432678,
                5.1554341,
                5.31654141,
                5.47764873,
                5.63875604,
                5.79986336,
                5.96097068,
                6.12207799,
                6.28318531,
            ]
        ),
        "shape": np.array([1.0, 2.0, 3.0]),
        "color": np.array([1.0]),
    }

    def __init__(self, root=os.path.join(DIR, "../data/dsprites/"), **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        dataset_zip = np.load(self.train_data)
        self.imgs = dataset_zip["imgs"]
        self.lat_values = dataset_zip["latents_values"]

    def download(self):
        """Download the dataset."""
        os.makedirs(self.root)
        subprocess.check_call(
            ["curl", "-L", type(self).urls["train"], "--output", self.train_data]
        )

    def __getitem__(self, idx):
        """Get the image of `idx`
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.
        lat_value : np.array
            Array of length 6, that gives the value of each factor of variation.
        """
        # stored image have binary and shape (H x W) so multiply by 255 to get pixel
        # values + add dimension
        sample = np.expand_dims(self.imgs[idx] * 255, axis=-1)

        # ToTensor transforms numpy.ndarray (H x W x C) in the range
        # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        sample = self.transforms(sample)

        lat_value = self.lat_values[idx]
        return sample, lat_value


class ECG5000(Dataset):
    def __init__(
        self,
        dir: pathlib.Path,
        train: bool = True,
        random_seed: int = 42,
        experiment: str = "features",
    ):
        if experiment not in ["features", "examples"]:
            raise ValueError("The experiment name is either features or examples.")
        self.dir = dir
        self.train = train
        self.random_seed = random_seed
        if not dir.exists():
            os.makedirs(dir)
            self.download()

        # Load the data and create a train/test set with split
        with open(self.dir / "ECG5000_TRAIN.arff") as f:
            data, _ = loadarff(f)
            total_df = pd.DataFrame(data)
        with open(self.dir / "ECG5000_TEST.arff") as f:
            data, _ = loadarff(f)
            total_df = total_df.append(pd.DataFrame(data))

        # Isolate the target column in the dataset
        label_normal = b"1"
        new_columns = list(total_df.columns)
        new_columns[-1] = "target"
        total_df.columns = new_columns

        if experiment == "features":
            # Split the dataset in normal and abnormal examples
            normal_df = total_df[total_df.target == label_normal].drop(
                labels="target", axis=1
            )
            anomaly_df = total_df[total_df.target != label_normal].drop(
                labels="target", axis=1
            )
            if self.train:
                df = normal_df
            else:
                df = anomaly_df
            labels = [int(self.train) for _ in range(len(df))]

        elif experiment == "examples":
            df = total_df.drop(labels="target", axis=1)
            labels = [0 if label == label_normal else 1 for label in total_df.target]

        else:
            raise ValueError("Invalid experiment name.")

        sequences = df.astype(np.float32).to_numpy().tolist()
        sequences = [
            torch.tensor(sequence).unsqueeze(1).float() for sequence in sequences
        ]
        self.sequences = sequences
        self.labels = labels
        self.n_seq, self.seq_len, self.n_features = torch.stack(sequences).shape

    def __len__(self):
        return self.n_seq

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

    def download(self):
        """Download the dataset."""
        url = "http://timeseriesclassification.com/Downloads/ECG5000.zip"
        logging.info("Downloading the ECG5000 Dataset.")
        data_zip = self.dir / "ECG5000.zip"
        wget.download(url, str(data_zip))
        with ZipFile(data_zip, "r") as zip_ref:
            zip_ref.extractall(self.dir)
        logging.info("Finished Downloading.")


class MaskedMNIST(MNIST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        masks: torch.Tensor = None,
    ):
        super().__init__(root, train=train, download=True)
        self.masks = masks

    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        image = self.masks[index] * image
        return image, target


class CIFAR10Pair(CIFAR10):
    """Generate mini-batche pairs on CIFAR10 training set."""

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)  # .convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        return torch.stack(imgs), target  # stack a positive pair
