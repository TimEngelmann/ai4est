from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def create_split_dataframe(path: str, data:pd.DataFrame, method:str, splits):
    """
    path: to the images
    method: "across_sites" or "by_site"
    data: contains file locations, labels and additional info
    splits: [train, val, test],
            summing up to 1 for method "across_sites"
            summing up to 6 for method "by_site"
    """
    sites = ['Carlos Vera Arteaga RGB', 'Carlos Vera Guevara RGB',
             'Flora Pluas RGB', 'Leonor Aspiazu RGB', 'Manuel Macias RGB',
             'Nestor Macias RGB']
    
    train_dataset = pd.DataFrame(data=None, columns=data.columns)
    val_dataset = pd.DataFrame(data=None, columns=data.columns)
    test_dataset = pd.DataFrame(data=None, columns=data.columns)

    if method=="across_sites":
        ptrain, pval, ptest=splits
        assert(ptrain+pval+ptest==1)
        for i in range(len(sites)):
            data_site = data.loc[data['site']==sites[i]]
            train_site, test_val_site = train_test_split(data_site, train_size=ptrain)
            val_site, test_site= train_test_split(test_val_site, train_size=pval/(pval+ptest))
            train_dataset = pd.concat([train_dataset, train_site])
            val_dataset = pd.concat([val_dataset, val_site])
            test_dataset = pd.concat([test_dataset, test_site])

    if method=="by_site":
        ntrain, nval, ntest=splits
        assert(ntrain+nval+ntest==6)
        idx=np.arange(6)
        np.random.shuffle(idx)
        print(idx)
        for i in idx[0:ntrain]:
            data_site = data.loc[data['site'] == sites[i]]
            train_dataset = pd.concat([train_dataset, data_site])
        for i in idx[ntrain:(ntrain+nval)]:
            data_site = data.loc[data['site'] == sites[i]]
            val_dataset = pd.concat([val_dataset, data_site])
        for i in idx[(ntrain+nval):6]:
            data_site = data.loc[data['site'] == sites[i]]
            test_dataset = pd.concat([test_dataset, data_site])
    
    train_dataset = train_dataset.reset_index(drop=True)
    val_dataset = val_dataset.reset_index(drop=True)
    test_dataset = test_dataset.reset_index(drop=True)
    
    return train_dataset, val_dataset, test_dataset


class PatchesDataSet(Dataset):
    def __init__(self, path, df, transform=None):
        """
        Args:
            path (string): Path to the directory with the site images
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.path = path
        # "/Users/victoriabarenne/Documents/ai4good2/dataset/"
        self.df = df
        self.transform = transform

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
        i, j = self.df["site index"][item]
        
        image = torch.load(path)[:, i, j, ...]
        carbon = self.df["carbon"][item]

        assert image.shape[0] == 3
        assert image.ax


        if self.transform is not None:
            image = self.transform(image)

        return image, carbon, site, rotation


def train_val_test_dataset(path: str, data:pd.DataFrame, method:str, splits):
    train, val, test= create_split_dataframe(path, data, method, splits)
    train_dataset= PatchesDataSet(path, train)
    val_dataset = PatchesDataSet(path, val)
    test_dataset = PatchesDataSet(path, test)
    
    return train_dataset, val_dataset, test_dataset


def train_val_test_dataloader(path:str, data:pd.DataFrame, method:str, splits, batch_size):
    train_dataset, val_dataset, test_dataset= train_val_test_dataset(path, data, method, splits)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader, test_loader