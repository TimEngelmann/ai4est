from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from torchvision.transforms import ToTensor
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

def create_patches_dataframe(path):
    files = list(Path(path).glob('**/*.npz'))
    df = list()

    for i in range(len(files)):
        print(i)
        data = np.load(str(files[i]))
        path_to_img=files[i]
        carbon = data["carbon"]
        site = data["site"].item()
        rotation = data["rotation"].item()
        idx = 0
        for key in list(data.keys()):
            if (key != "carbon") & (key != "site") & (key != "rotation"):
                #image = data[key][0:3, :, :]
                name= key
                new = {"path": path_to_img, "name": name, "carbon": carbon[idx], "site": site, "rotation": rotation}
                df.append(new)
                idx += 1
    return pd.DataFrame(df)
def create_split_dataframe(path: str, method:str, splits):
    """
    path: to the images
    method: "across_sites" or "by_site"
    splits: [train, val, test],
            summing up to 1 for method "across_sites"
            summing up to 6 for method "by_site"
    """
    sites = ['Carlos Vera Arteaga RGB', 'Carlos Vera Guevara RGB',
             'Flora Pluas RGB', 'Leonor Aspiazu RGB', 'Manuel Macias RGB',
             'Nestor Macias RGB']
    patches = create_patches_dataframe(path)
    train_dataset = pd.DataFrame(data=None, columns=patches.columns)
    val_dataset = pd.DataFrame(data=None, columns=patches.columns)
    test_dataset = pd.DataFrame(data=None, columns=patches.columns)

    if method=="across_sites":
        ptrain, pval, ptest=splits
        assert(ptrain+pval+ptest==1)
        for i in range(len(sites)):
            patches_site=patches.loc[patches['site']==sites[i]]
            train_site, test_val_site = train_test_split(patches_site, train_size=ptrain)
            val_site, test_site= train_test_split(test_val_site, train_size=pval/(pval+ptest))
            train_dataset= pd.concat([train_dataset, train_site])
            val_dataset = pd.concat([val_dataset, val_site])
            test_dataset = pd.concat([test_dataset, test_site])

    if method=="by_site":
        ntrain, nval, ntest=splits
        assert(ntrain+nval+ntest==6)
        idx=np.arange(6)
        np.random.shuffle(idx)
        print(idx)
        for i in idx[0:ntrain]:
            patches_site = patches.loc[patches['site'] == sites[i]]
            train_dataset = pd.concat([train_dataset, patches_site])
        for i in idx[ntrain:(ntrain+nval)]:
            patches_site = patches.loc[patches['site'] == sites[i]]
            val_dataset = pd.concat([val_dataset, patches_site])
        for i in idx[(ntrain+nval):6]:
            patches_site = patches.loc[patches['site'] == sites[i]]
            test_dataset = pd.concat([test_dataset, patches_site])
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
        self.files = list(Path(self.path).glob('**/*.npz'))
        self.df = df
        self.transform= transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        name=self.df["name"][item]
        path=self.df["path"][item]
        carbon = self.df["carbon"][item]
        site = self.df["site"][item]
        rotation = self.df["rotation"][item]
        image= np.load(path)[name][0:3, :, :]
        image = np.moveaxis(image, 0, -1)
        if self.transform is not None:
            image = self.transform(image)
        return image, carbon, site, rotation
def train_val_test_dataset(path: str, method:str, splits, transform=ToTensor()):
    train, val, test= create_split_dataframe(path, method, splits)
    train_dataset= PatchesDataSet(path, train, transform)
    val_dataset = PatchesDataSet(path, val, transform)
    test_dataset = PatchesDataSet(path, test, transform)
    return train_dataset, val_dataset, test_dataset
def train_val_test_dataloader(path:str, method:str, splits, batch_size, transform=ToTensor()):
    train_dataset, val_dataset, test_dataset= train_val_test_dataset(path, method, splits, transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, test_loader

