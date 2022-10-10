import pandas as pd
import numpy as np

'''
input
- patches: pandas df [site, (x1, y1), (x2, y2), (x3, y3), (x4, y4)]
- trees: pandas df [site, x, y, lat, lon, carbon]
- gps_error: {site: [lon_error, lat_error]}

output
- pandas df [site, (x1, y1), (x2, y2), (x3, y3), (x4, y4), carbon]

'''

def estimate_agb(patches, trees, gps_error):

    # convert gps_error in xy_error
    xy_error = {}
    for site in gps_error.keys():
        trees_site = trees[trees.site == site].sort_values(by="lon")
        coeff_x = np.polyfit(trees_site.lon, trees_site.X, 1)
        trees_site = trees_site.sort_values(by="lat")
        coeff_y = np.polyfit(trees_site.lat, trees_site.Y, 1)
        xy_error[site] = [abs(coeff_x[0] * gps_error[site][0]), abs(coeff_y[0] * gps_error[site][1])]

    # only works for non rotated triangles
    def in_rectangle(point, vertices):
        if point[0] >= vertices[0][0] and point[0] < vertices[1][0] and point[1] >= vertices[0][1] and point[1] < vertices[3][1]:
            return True
        return False

    # for all patches sum over all trees in patch
    carbon_patches = []
    tree_patch = {}
    for idx_patch, patch in patches.iterrows():
        carbon_patch = 0
        vertices = np.array([patch.a, patch.b, patch.c, patch.d])

        for idx_tree, tree in trees[trees.site == patch.site].iterrows():
            point = np.array((tree.X, tree.Y))
            if in_rectangle(point, vertices):
                carbon_patch += tree.carbon
                tree_patch[idx_tree] = idx_patch
        carbon_patches.append(carbon_patch)

    patches['carbon'] = carbon_patches
    return patches


# Testing function

# import trees
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