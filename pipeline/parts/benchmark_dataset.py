import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import torch
from IPython import embed
import logging
from torchvision.transforms import ToTensor, Compose, CenterCrop, Pad, Resize
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from parts.data_split import create_split_dataframe

def create_benchmark_dataset(paths):
    final_df= pd.read_csv(paths["reforestree"]+"mapping/final_dataset.csv")
    my_df= final_df[['img_name', 'Xmin', 'Ymin', 'Xmax', 'Ymax', 'AGB', 'carbon']]
    sites= np.unique(final_df["img_name"])

    # Creating the tree crown images for all bounding boxes
    logging.info("Creating unfiltered tree crown dataframe")
    tree_crowns = pd.DataFrame([] ,columns=["tree_img", "coord", "site", "carbon"])

    for site in sites:
        df_site= my_df[my_df['img_name']==site]
        df_site= df_site.reset_index(drop=True)
        image_site= cv2.imread(paths["dataset"] + "sites/" + f"{site}_image.png")

        for i in range(len(df_site)):
            tree_coord= np.array(df_site.iloc[i][['Xmin', 'Ymin', 'Xmax', 'Ymax']]).astype(int)
            carbon= df_site.iloc[i]['carbon'].item()
            tree_img= image_site[tree_coord[1]:tree_coord[3], tree_coord[0]:tree_coord[2], :]
            new_df=pd.DataFrame([[tree_img, tree_coord, site, carbon]], columns=tree_crowns.columns)
            tree_crowns=pd.concat([tree_crowns, new_df], ignore_index=True)

    # Filtering the ones out of the sites boundaries
    logging.info("Filtering out of boundary tree crown bounding boxes")
    white_percentage= np.zeros(len(tree_crowns))
    for i in range(len(tree_crowns)):
        tree_img= tree_crowns.iloc[i]["tree_img"]
        surface= tree_img.shape[0]*tree_img.shape[1]
        if surface>0:
            white_percentage[i]= np.sum(np.all(tree_img==0, axis=2))/surface
        elif surface==0:
            white_percentage[i]=1

    white_threshold= 0.8
    tree_crowns_filtered= tree_crowns.iloc[white_percentage<white_threshold]
    tree_crowns_filtered= tree_crowns_filtered.reset_index(drop=True)
    logging.info(f"Only {len(tree_crowns_filtered)}/{len(tree_crowns)} of the bounding boxes were inside the site boundaries")

    # Padding/Cropping the images so that they are all of size 800x800 (as in the paper)
    transform= Compose([ToTensor(), CenterCrop(800)])
    tree_crowns_final=pd.DataFrame([], columns=["tree_img", "site", "carbon"])

    for i in range(len(tree_crowns_filtered)):
        tree_img= tree_crowns_filtered.iloc[i]["tree_img"]
        transformed_img= transform(tree_img)
        site = tree_crowns_filtered["site"][i]
        carbon= tree_crowns_filtered["carbon"][i]

        new_df=pd.DataFrame([[transformed_img, site, carbon]], columns=tree_crowns_final.columns)
        tree_crowns_final=pd.concat([tree_crowns_final, new_df], ignore_index=True)

    assert(len(tree_crowns_final)== len(tree_crowns_filtered))
    return tree_crowns_final

# TreeCrown torch Dataset
class TreeCrown(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform= transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        """
        Loading samples based on the information contained
        in self.df.
        """
        site = self.df["site"][item]
        image =  self.df["tree_img"][item]
        carbon = self.df["carbon"][item]


        if self.transform is not None:
            image = self.transform(image)

        return image, carbon, site, 0

def train_val_test_dataset_benchmark(data:pd.DataFrame, splits, transform=None):
    train, val, test= create_split_dataframe("", data, splits)
    train_dataset= TreeCrown(train, transform)
    val_dataset = TreeCrown(val, transform)
    test_dataset = TreeCrown(test, transform)

    return train_dataset, val_dataset, test_dataset

def train_val_test_dataloader_benchmark(data: pd.DataFrame, splits, batch_size, transform=None):
    train_dataset, val_dataset, test_dataset= train_val_test_dataset_benchmark(data, splits, transform)
    assert(len(train_dataset))>0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if len(val_dataset)>0:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    else:
        val_loader= None
    if len(test_dataset)>0:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    else:
        test_loader= None

    return train_loader, val_loader, test_loader


