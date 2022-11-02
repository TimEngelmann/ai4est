import torch
from torchvision.transforms.functional import rotate
import numpy as np
import pandas as pd

def get_upper_left(patch_size, img_shape):
    raise NotImplementedError()


def process_site(df, hyperparameters, paths, site):
    """
    For each sites rotates the image data and splits into
    patches stored in a single array. For each site and rotation
    angle, the 4-channel image is rotated and then reshaped into
    a Tensor of shape 1 x nrows x ncols x patch_size x patch_size,
    where nrows is the number of patches in each row and ncols is
    the number of patches in each col.
    """

    site_data = torch.from_numpy(np.load(paths["dataset"] + f"{site}"))
    
    rotations = hyperparameters["rotations"]
    patch_size = hyperparameters["patch_size"]
    
    df_prototype = pd.DataFrame([], columns=df.columns)
    df_prototype["upper left"] = get_upper_left(patch_size)
    df_prototype["site"] = site
    df_prototype["rotation"] = 0
   
    new_df = pd.DataFrame([], columns=df.columns)

    for angle in rotations:
        df_angle = df_prototype.copy()
        df_angle["angle"] = angle

        if angle != 0.0:
            site_data = rotate(site_data, angle)
    
        patched_data = site_data.unfold(1, patch_size,  patch_size).unfold(2, patch_size, patch_size)

        #TODO Compute carbon
        
        
        torch.save(patched_data, paths["dataset"] + f"processed/{site}_{angle}")
        new_df = pd.concat((new_df, df_angle))



    return new_df

def process(sites, hyperparameters, paths):
    raise NotImplementedError()

    df = pd.DataFrame([], columns=["carbon", "site", "rotation", "upper_left", "patch_size"])

    for site in sites:
        df = process_site(df, hyperparameters, paths, site)

    return df
