from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from torchvision.transforms import ToTensor
import torchvision.io
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

def create_split_dataframe(path: str, data:pd.DataFrame, splits):

    """
    Creating a dataframe and splitting it into training, , validating and testing
    """

    train_dataset = pd.DataFrame(data=None, columns=data.columns)
    val_dataset = pd.DataFrame(data=None, columns=data.columns)
    test_dataset = pd.DataFrame(data=None, columns=data.columns)
    train_val_test= np.zeros(len(data))

    for i in range(len(data)):
        if data["site"][i] in splits["training"]:
            train_val_test[i]= 1
        if data["site"][i] in splits["testing"]:
            train_val_test[i]=2
        if data["site"][i] in splits["validation"]:
            train_val_test[i] = 3

    train_dataset= data.loc[train_val_test==1]
    test_dataset= data.loc[train_val_test==2]
    val_dataset= data.loc[train_val_test==3]

    train_dataset = train_dataset.reset_index(drop=True)
    val_dataset = val_dataset.reset_index(drop=True)
    test_dataset = test_dataset.reset_index(drop=True)

    return train_dataset, val_dataset, test_dataset

class PatchesDataSet(Dataset):
    def __init__(self, path_to_dataset, df, transform=None):

        self.path = path_to_dataset
        self.df = df
        self.transform= transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        """
        Loading samples based on the information contained
        in self.df.
        """
        path = self.path + self.df["path"][item]
        site = self.df["site"][item]
        rotation = self.df["rotation"][item]

        image = torchvision.io.read_image(path)
        carbon = self.df["carbon"][item]

        assert image.shape[0] == 3


        if self.transform is not None:
            image = self.transform(image.double())

        return image, carbon, site, rotation


def train_val_test_dataset(path: str, data:pd.DataFrame, splits, transform):
    """
    Creates training, validating and testing datasets
    """
    train, val, test= create_split_dataframe(path, data, splits)
    train_dataset= PatchesDataSet(path, train, transform)
    val_dataset = PatchesDataSet(path, val, transform)
    test_dataset = PatchesDataSet(path, test, transform)
    return train_dataset, val_dataset, test_dataset

def train_val_test_dataloader(path: str, data: pd.DataFrame, splits, batch_size, transform=None):
    """
    Creates training, validating and testing dataloaders
    """
    train_dataset, val_dataset, test_dataset= train_val_test_dataset(path, data, splits, transform)
    assert(len(train_dataset))>0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if len(val_dataset)>0:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader= None
    if len(test_dataset)>0:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        test_loader= None

    return train_loader, val_loader, test_loader


def compute_mean(hyperparameters:dict, data:pd.DataFrame, path:str, benchmark_dataset:bool):
    if not benchmark_dataset:
        """
        Compute the mean and std over the entire dataset. 
        """
        patch_size = hyperparameters["patch_size"]

        dataset = PatchesDataSet(path, data)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        mean = torch.zeros(3)
        std = torch.zeros(3)
        count = len(data)  * patch_size ** 2
        for batch in dataloader:
            imgs = batch[0]
            batch_samples = imgs.size(0)
            imgs = imgs.view(batch_samples, 3, -1) / 256
            mean = mean + imgs.sum((0,-1))

        mean = mean * 256 / count

        for batch in dataloader:
            imgs = batch[0]
            batch_samples = imgs.size(0)
            imgs = imgs.view(batch_samples, 3, -1)
            imgs = torch.moveaxis(imgs, 1, -1)
            std = std + ((imgs - mean) ** 2).sum((0,1))

        std = torch.sqrt((std / count)) 
    
    else:
        dataloader = DataLoader(
            data.tree_img.values,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

        mean = torch.zeros(3)
        std = torch.zeros(3)
        count = 0
        for batch in dataloader:
            imgs = batch[0]
            batch_samples = imgs.size(0)
            imgs = imgs / 256
            mean = mean + imgs.sum((1,2))
            count += imgs.shape[1] * imgs.shape[2]

        mean = mean * 256 / count

        for batch in dataloader:
            imgs = batch[0]
            batch_samples = imgs.size(0)
            std = std + ((torch.moveaxis(imgs, 0, -1) - mean) ** 2).sum((0,1))

        std = torch.sqrt((std / count)) 

    return mean, std
