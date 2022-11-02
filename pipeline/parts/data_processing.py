import torch
from torchvision.transforms.functional import rotate
import numpy as np
import pandas as pd


def process_site(df, hyperparameters, paths, site):
    raise NotImplementedError()

    site_data = torch.from_numpy(np.load(paths["dataset"] + f"{site}"))
    df_site = pd.DataFrame([], columns=df.columns)

    rotations = hyperparameters["rotations"]
    patch_size = hyperparameters["patch_size"]

    for angle in rotations:
        if angle != 0.0:
            site_data = rotate(site_data, angle)

        #TODO make patches

        torch.save(processed_data, paths["dataset"] + f"processed/{site}_{angle}")

    return new_df

def process(sites, hyperparameters, paths):
    raise NotImplementedError()

    df = pd.DataFrame([], columns=["carbon", "site", "rotation", "upper_left", "patch_size"])

    for site in sites:
        df = process_site(df, hyperparameters, paths, site)

    return df
