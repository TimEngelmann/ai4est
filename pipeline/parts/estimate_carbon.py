import pandas as pd
import numpy as np

from scipy.stats import multivariate_normal

from helper.constants import get_gps_error

'''
input
- patches: pandas df [site, [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]]
- trees: pandas df [site, x, y, lat, lon, carbon]
- gps_error: {site: [lon_error, lat_error]}

output
- pandas df [site, (x1, y1), (x2, y2), (x3, y3), (x4, y4), carbon]

'''

def estimate_carbon(patches, trees, gps_error):

    # calculate carbon distributions with gaussian distribution
    carbon_distributions = {}
    for site in gps_error.keys():
        sigma_multiple = 10
        max_x_tree = int(sigma_multiple * np.sqrt(gps_error[site][0]))
        max_y_tree = int(sigma_multiple * np.sqrt(gps_error[site][1]))
        y_range, x_range = np.mgrid[0:max_y_tree, 0:max_x_tree]
        pos = np.dstack((y_range, x_range))
        rv = multivariate_normal([max_y_tree/2, max_x_tree/2], [[gps_error[site][1], 0], [0, gps_error[site][0]]])
        gaussian = rv.pdf(pos)

        padding = 300
        trees_site = trees[trees.site == site]
        max_x = int(np.max(trees_site.X) + padding)
        max_y = int(np.max(trees_site.Y) + padding)
        carbon_distribution = np.zeros((max_y, max_x))

        for idx, tree in trees_site[trees_site.carbon < 50].iterrows():
            gaussian_tree = gaussian * tree.carbon

            start_x = int(tree.X - max_x_tree/2)
            start_y = int(tree.Y - max_y_tree/2)
            end_x = int(tree.X + max_x_tree/2)
            end_y = int(tree.Y + max_y_tree/2)
            if (start_x < 0):
                gaussian_tree = gaussian_tree[:, abs(start_x):]
            if (start_y < 0):
                gaussian_tree = gaussian_tree[abs(start_y):, :]
            if (end_x > max_x):
                gaussian_tree = gaussian_tree[:, :max_x_tree - (end_x - max_x)]
            if (end_y > max_y):
                gaussian_tree = gaussian_tree[:max_y_tree - (end_y - max_y), :]

            carbon_distribution[max(start_y, 0):min(end_y, max_y), max(start_x, 0):min(end_x, max_x)] += gaussian_tree
        carbon_distributions[site] = carbon_distribution

    # only works for non rotated triangles
    carbon_patches = []
    for idx_patch, patch in patches.iterrows():
        window = carbon_distributions[patch.site][patch.vertices[0][1]:patch.vertices[2][1], patch.vertices[0][0]:patch.vertices[2][0]]
        carbon_patch = np.sum(window)
        carbon_patches.append(carbon_patch)

    patches['carbon'] = carbon_patches
    return patches

if __name__ == "__main__":
    trees = pd.read_csv('data/reforestree/field_data.csv')
    trees = trees[["site", "X", "Y", "lat", "lon", "carbon"]]

    # gps_error: {site: [lon_error, lat_error]}
    gps_error = get_gps_error()

    # create dummy patches
    patch_size = 2000
    patches_array = []
    for site in gps_error.keys():
        for x in range(3):
            for y in range(3):
                vertices = np.array([(x,y), (x+1,y), (x+1,y+1), (x,y+1)]) * patch_size
                patches_array.append([site, vertices])
    patches = pd.DataFrame(patches_array, columns=["site", "vertices"])
    # run function
    patches_with_label = estimate_carbon(patches, trees, gps_error)
    print(patches_with_label)