import pandas as pd
import numpy as np

'''
input
- patches: pandas df [site, [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]]
- trees: pandas df [site, x, y, lat, lon, carbon]
- gps_error: {site: [lon_error, lat_error]}

output
- pandas df [site, (x1, y1), (x2, y2), (x3, y3), (x4, y4), carbon]

'''

def estimate_agb(patches, trees, gps_error):

    # calculate carbon distributions with point weights
    padding = 500
    carbon_distributions = {}
    for site in patches.site.unique():
        max_x = int(np.max(trees.X) + padding)
        max_y = int(np.max(trees.Y) + padding)

        carbon_distribution = np.zeros((max_y, max_x))
        trees_site = trees[trees.site == site]
        for x, y, carbon in zip(trees_site.X, trees_site.Y, trees_site.carbon):
            carbon_distribution[int(y), int(x)] = carbon
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

    # import gps error
    gps_error = {"Flora Pluas RGB": [0.25, 0.66],
                "Nestor Macias RGB": [0.6, 0.53],
                "Manuel Macias RGB": [0.69, 0.30],
                "Leonor Aspiazu RGB": [0.47, 0.45],
                "Carlos Vera Arteaga RGB": [0.26, 0.59],
                "Carlos Vera Guevara RGB": [0.27, 0.65]}

    # create dummy patches
    patch_size = 2000
    patches_array = []
    for site in gps_error.keys():
        for x in range(3):
            for y in range(3):
                coordinates = np.array([(x,y), (x+1,y), (x+1,y+1), (x,y+1)]) * patch_size
                patches_array.append([site, coordinates[0], coordinates[1], coordinates[2], coordinates[3]])
    patches = pd.DataFrame(patches_array, columns=["site", "a", "b", "c", "d"])
    # run function
    patches_with_label = estimate_agb(patches, trees, gps_error)
    print(patches_with_label)