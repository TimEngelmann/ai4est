import numpy as np
import cv2
import pandas as pd
from IPython import embed
import logging
from torchvision.transforms import ToTensor, Compose, CenterCrop, Pad, Resize
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from parts.data_split import create_split_dataframe
import ot


def greedy_matching(otplan):
    """
    Greedily chooses best match and removes the 
    corresponding source and target from the optimal 
    transport plan to get a one-to-one matching.
    """

    tmp_plan = otplan.copy()
    matches = np.empty(tmp_plan.shape[0], dtype=int)

    while tmp_plan.sum() != 0:
        i, j = np.unravel_index(tmp_plan.argmax(), tmp_plan.shape) #finding best match
        matches[i] = j
        
        #removing match from optimal transport plan
        tmp_plan[i, :] = 0.
        tmp_plan[:, j] = 0.

    return matches


def matching_site(bboxes_site, field_data_site, method):
    """
    Matching bounding boxes to field data points using 
    optimal transport and a greedy matching strategy.
    The optimal transport provideds the most likely
    matches for each bounding box and the greedy matching
    is used to get a one-to-one mapping between the two.
    Inputs:
        bboxes : dataframe
            stores the center point of each bounding box
            for a particular site
        field_data: dataframe
            contains the field data points with pixel
            coordinates for a particular site
        method : string
            which method to use for the optimal transport
            plan computation. Currently only the sinkhorn
            and emd algorithms are supported.
    Returns:
        dataframe : bboxes dataframe with additional column
            containing the indices of the matched field
            data points for a single site
    """
    idxs = field_data_site.index

    xbb = bboxes_site[["Xcenter", "Ycenter"]].values
    xf = field_data_site[["X", "Y"]].values

    normalizer = np.max(xbb, axis=0)
    M_norm = ot.dist(xbb / normalizer, xf / normalizer)

    a, b = np.ones((xbb.shape[0],)) / xbb.shape[0], np.ones((xf.shape[0],)) / xf.shape[0]  # uniform distribution on samples

    if method == "sinkhorn":
        otplan = ot.sinkhorn(a, b, M_norm, reg=0.1, numItermax=100000)
    elif method == "emd":
        otplan = ot.emd(a, b, M_norm, numItermax=1e5)
    else:
        raise NotImplementedError(f"Matching method {method} is not implemented")
    
    matches = greedy_matching(otplan)
    bboxes_site["matches"] = idxs[matches]

    return bboxes_site    


def matching(bboxes, field_data, method="emd"):
    """
    Matching bounding boxes to field data points using 
    optimal transport and a greedy matching strategy.
    The optimal transport provideds the most likely
    matches for each bounding box and the greedy matching
    is used to get a one-to-one mapping between the two.
    Inputs:
        bboxes : dataframe
            stores the center point of each bounding box
            for all sites
        field_data: dataframe
            contains the field data points with pixel
            coordinates for all sites
        method : string
            which method to use for the optimal transport
            plan computation. Currently only the sinkhorn
            and emd algorithms are supported, the default
            is sinkhorn.
    Returns:
        dataframe : bboxes dataframe with additional column
            containing the indices of the matched field
            data points and their correspondign carbon
            values
    """
    sites = field_data.site.unique()

    matched_df = pd.DataFrame([])
    for site in sites:
        bboxes_site = bboxes[bboxes.site == site].copy()
        field_data_site = field_data[field_data.site == site]
        df_site = matching_site(bboxes_site, field_data_site, method)
        matched_df = pd.concat([matched_df, df_site])

    #using matching to determine carbon
    matched_df["carbon"] = field_data.iloc[matched_df.matches].carbon

    return matched_df

def create_benchmark_dataset(paths):
    columns = ['img_name', 'Xmin', 'Ymin', 'Xmax', 'Ymax']
    tree_crowns = pd.read_csv(paths["reforestree"] + "mapping/final_dataset.csv", usecols=columns)
    tree_crowns.rename(columns={"img_name": "site"}, inplace=True)

    field_data = pd.read_csv(paths["reforestree"] + "field_data.csv")

    sites = field_data.site.unique()

    #Computing center points of bboxes
    tree_crowns["Xcenter"] = (tree_crowns.Xmin + tree_crowns.Xmax) / 2
    tree_crowns["Ycenter"] = (tree_crowns.Ymin + tree_crowns.Ymax) / 2

    
    # Creating the tree crown images for all bounding boxes
    tree_images = np.empty(len(tree_crowns), dtype=object)
    for site in sites:
        image_site = cv2.imread(paths["dataset"] + "sites/" + f"{site}_image.png")
        for i in tree_crowns[tree_crowns.site == site].index:
            xmin, xmax = tree_crowns.loc[i, ["Xmin", "Xmax"]].astype(int)
            ymin, ymax = tree_crowns.loc[i, ["Ymin", "Ymax"]].astype(int)
            tree_images[i] = image_site[ymin:ymax, xmin:xmax, :]
    
    tree_crowns["tree_img"] = tree_images

    # Filtering the ones out of the sites boundaries
    logging.info("Filtering out of boundary tree crown bounding boxes")
    white_calculator = lambda img : np.mean((img == 0.0).all(axis=2))
    white_percentage = tree_crowns["tree_img"].apply(white_calculator)

    white_threshold = 0.8
    tree_crowns_filtered = tree_crowns.loc[white_percentage<white_threshold]
    tree_crowns_filtered = tree_crowns_filtered.reset_index(drop=True)
    logging.info(f"Only {len(tree_crowns_filtered)}/{len(tree_crowns)} of the bounding boxes were inside the site boundaries")

    #Adding carbon values from matched field data
    matched_df = matching(tree_crowns_filtered, field_data)

    # Padding/Cropping the images so that they are all of size 800x800 (as in the paper)
    transform = Compose([ToTensor(), CenterCrop(800)])
    matched_df["tree_img"].apply(transform)
    matched_df.to_csv(paths["dataset"] + "benchmark_dataset.csv")
    return matched_df

# TreeCrown torch Dataset
class TreeCrown(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

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


if __name__ == "__main__":
    paths = {
        "reforestree" : "/Users/jan/ai4good/data/reforestree/",
        "dataset" : "/Users/jan/ai4good/data/dataset/"
    }
    df = create_benchmark_dataset(paths)

    df.to_csv("matching.csv", columns=["Xcenter", "Ycenter","matches"])