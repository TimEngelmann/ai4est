import numpy as np
from scipy.stats import multivariate_normal

'''
input
- patches: pandas df [site, [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]]
- trees: pandas df [site, x, y, lat, lon, carbon]
- gps_error: {site: [lon_error, lat_error]}

output
- pandas df [site, (x1, y1), (x2, y2), (x3, y3), (x4, y4), carbon]

'''

def compute_carbon_distribution(site, img_shape, trees, gps_error):

    # calculate carbon distributions with gaussian distribution
    sigma_multiple = 10
    max_x_tree = int(sigma_multiple * np.sqrt(gps_error[site][0]))
    max_y_tree = int(sigma_multiple * np.sqrt(gps_error[site][1]))
    y_range, x_range = np.mgrid[0:max_y_tree, 0:max_x_tree]
    pos = np.dstack((y_range, x_range))
    rv = multivariate_normal([max_y_tree/2, max_x_tree/2], [[gps_error[site][1], 0], [0, gps_error[site][0]]])
    gaussian = rv.pdf(pos)
    padding = 300
    trees_site = trees[trees.site == site]
    max_x = img_shape[2]
    max_y = img_shape[1]
    carbon_distribution = np.zeros((img_shape[1], img_shape[2]))
    for _, tree in trees_site[trees_site.carbon < 50].iterrows():
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

    #normalizing distribution so that total carbon mass agrees with field data
    total_carbon_site = trees_site["carbon"].sum()
    total_carbon_distribution = carbon_distribution.sum()

    return carbon_distribution * total_carbon_site / total_carbon_distribution