import cv2
from matplotlib import patches
import seaborn as sns
import pandas as pd
import numpy as np

from parts.patches import make_grid

# hyperparameters
patch_size = 400

trees = pd.read_csv('data/reforestree/field_data.csv')
trees = trees[["site", "X", "Y", "lat", "lon", "carbon"]]

for site in trees.site.unique():

    path = 'data/reforestree/wwf_ecuador/RGB Orthomosaics/{}.tif'.format(site)
    img = cv2.imread(path)
    img = np.array(img)

    grid = make_grid(img.shape, patch_size)

    print(grid)

