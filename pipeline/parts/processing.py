import os
import glob
import logging
import torch
from torchvision.transforms.functional import rotate
from torchvision.io import read_image, write_png
import numpy as np
import pandas as pd


def get_upper_left(patch_size, img_shape):
    """
    Gets the upper left corner of each patch and the 
    corresponding index in the unfolded Tensor.
    """
    shape = np.array(img_shape[1:])
    npatches = shape // patch_size
    assert (shape % patch_size == 0).all()

    upper_left = np.zeros((npatches.prod(), 2))
    indices = np.zeros((npatches.prod(), 2))
    
    count = 0
    for i in range(npatches[0]):
        for j in range(npatches[1]):
            upper_left[count, :] = np.array([i * patch_size, j * patch_size])
            indices[count, :] = np.array([i, j], dtype=int)
            count += 1

    return upper_left, indices


def process_site(df, hyperparameters, paths, site):
    """
    For each sites rotates the image data and splits into
    patches stored in a single array. For each site and rotation
    angle, the 4-channel image is rotated and then reshaped into
    a Tensor of shape 1 x nrows x ncols x patch_size x patch_size,
    where nrows is the number of patches in each row and ncols is
    the number of patches in each col.
    """
    logging.info("Processing data for site %s", site)
    
    # load image and carbon distribution and place them into site_data
    img_0 = read_image(paths["dataset"] + "sites/" + f"{site}_image.png")[:3, ]
    carbon_0 = torch.from_numpy(np.load(paths["dataset"] + "sites/" + f"{site}_carbon.npy"))
    carbon_0 = carbon_0.view(1, carbon_0.shape[0], carbon_0.shape[1])

    # start
    rotations = hyperparameters["rotations"]
    patch_size = hyperparameters["patch_size"]
    filter_threshold = hyperparameters["filter_threshold"]
    
    _, site_index = get_upper_left(patch_size, img_0.shape)
   
    paths_output = []
    for angle in rotations:
        logging.info("Processing data with rotation angle %s", angle)
        df_angle = pd.DataFrame([], columns=df.columns).copy()
        df_angle["site_index"] = pd.Series(((int(site_index[i,0]), int(site_index[i,1])) for i in range(site_index.shape[0])))
        df_angle["rotation"] = [angle] * len(df_angle) #specifying current angle
        df_angle["site"] = [site] * len(df_angle)
        df_angle["patch size"] = [patch_size] * len(df_angle)

        if angle != 0:
            carbon = rotate(carbon_0, angle)
            img = rotate(img_0, angle)
    
            patches = img.unfold(1, patch_size,  patch_size).unfold(2, patch_size, patch_size)
            carbon_patches = carbon.unfold(1, patch_size,  patch_size).unfold(2, patch_size, patch_size)
            df_angle["carbon"] = carbon_patches.sum(dim=(-1,-2)).reshape(-1)         
            del carbon_patches

        else:
            patches = img_0.unfold(1, patch_size,  patch_size).unfold(2, patch_size, patch_size)
            carbon_patches = carbon_0.unfold(1, patch_size,  patch_size).unfold(2, patch_size, patch_size)
            df_angle["carbon"] = carbon_patches.sum(dim=(-1,-2)).reshape(-1)         
            del carbon_patches

        #filtering empty patches
        logging.info("Filtering white patches")
        is_white = ((patches == 0).sum(dim=(0, -1,-2)) > filter_threshold * patch_size ** 2)
        filter_series = df_angle.apply(lambda row : not is_white[row["site_index"]], axis=1)
        filtered_df_angle = df_angle[filter_series]


        #testing if any of the removed patches have nonzero carbon
        if (df_angle.loc[~filter_series, "carbon"] != 0.0).any():
            #computing how many removed patches have positive carbon
            num_positive = (df_angle.loc[~filter_series, "carbon"] != 0.0).sum()
            num_removed = (~filter_series).sum()
            logging.warning("%f percent of removed patches have nonzero carbon", num_positive * 100 / num_removed)

            #finding maximum nonzero carbon
            max_removed_carbon = df_angle.loc[~filter_series, "carbon"].max()
            index = df_angle.loc[df_angle.index[df_angle["carbon"].argmax()], "site_index"]
            logging.warning("Max removed carbon is %f at index %s", max_removed_carbon, index)

        # torch.save(site_data, paths["dataset"] + site_angle_path)
        for idx, patch in filtered_df_angle.iterrows():
            i, j = patch["site_index"]
            image_path = "patches/" + f"{site}_{angle}_{idx}.png"
            filtered_df_angle.loc[idx, "path"] = image_path

            write_png(patches[:, i, j, ], paths["dataset"] + image_path)

        df = pd.concat((df, filtered_df_angle)) 

    return df

def process(sites, hyperparameters, paths):
    """
    Repeat the processing for each site and save the 
    relevant information in a DataFrame.
    """

    df = pd.DataFrame([], columns=["carbon", "path", "site", "rotation", "patch size", "site_index"])
    
    # creating folder "paths["dataset"]/patches" if it doesn't exist
    if not os.path.exists(paths["dataset"] + "patches"):
        logging.info("Creating directory %s", paths["dataset"] + "patches")
        os.makedirs(paths["dataset"] + "patches")

    #removing the files from previous processing run
    logging.info("Removing old data")
    files = glob.glob(paths["dataset"] + "patches/*.png")
    for f in files:
        os.remove(f)

    logging.info("Processing data")
    for site in sites:
        df = process_site(df, hyperparameters, paths, site)

    df.to_csv(paths["dataset"] + "patches_df.csv")

    #Make df index regular again 
    df = df.reset_index()
    return df
