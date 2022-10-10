import cv2
from matplotlib import patches
import seaborn as sns
import pandas as pd
import numpy as np

from parts.patches import save_patches

# hyperparameters
patch_size = 400

# paths
path_to_reforestree = "/data/reforestree/"
# TODO :  set path where patches will be saved
path_to_dataset = "./"

trees = pd.read_csv('data/reforestree/field_data.csv')
trees = trees[["site", "X", "Y", "lat", "lon", "carbon"]]

for site in trees.site.unique():
    save_patches(site, patch_size, path_to_reforestree, path_to_dataset)
