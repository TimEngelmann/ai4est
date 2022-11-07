import torch
from torchvision.transforms.functional import rotate
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

    upper_left = np.zeros((npatches.size, 2))
    indices = np.zeros((npatches.size, 2))
    
    count = 0
    for i in range(npatches[0]):
        for j in range(npatches[1]):
            upper_left[count, :] = np.array([i * patch_size, j * patch_size])
            indices[count, :] = np.array([i, j])
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

    site_data = torch.from_numpy(np.load(paths["dataset"] + f"{site}.npy"))
    
    rotations = hyperparameters["rotations"]
    patch_size = hyperparameters["patch_size"]
    
    new_df = pd.DataFrame([], columns=df.columns)

    #saving the data which stays the same for each rotation angle
    upper_left, site_index = get_upper_left(patch_size, site_data.shape)
   
    df_prototype = pd.DataFrame([], columns=df.columns)
    df_prototype["site index"] = site_index
    df_prototype["upper left"] = upper_left
    df_prototype["site"] = site
    df_prototype["patch size"] = patch_size
    
    for angle in rotations:
        df_angle = df_prototype.copy()
        df_angle["rotation"] = angle #specifying current angle
        df_angle["path"] = f"processed/{site}_{angle}.npy"

        if angle != 0.0:
            site_data = rotate(site_data, angle)
    
        patched_data = site_data.unfold(1, patch_size,  patch_size).unfold(2, patch_size, patch_size)
        df_angle["carbon"] = patched_data[-1,].sum(dim=(-1,-2)).reshape(-1)         
        
        torch.save(patched_data, paths["dataset"] + df_angle["path"])
        new_df = pd.concat((new_df, df_angle))

    return new_df

def process(sites, hyperparameters, paths):
    """
    Repeat the processing for each site and save the 
    relevant information in a DataFrame.
    """

    df = pd.DataFrame([], columns=["carbon", "path", "site", "rotation", "upper left", "patch size", "site index"])

    for site in sites:
        df = process_site(df, hyperparameters, paths, site)

    return df
