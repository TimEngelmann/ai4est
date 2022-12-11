import numpy as np
import logging
from scipy.stats import multivariate_normal


def compute_carbon_distribution(site, img_shape, trees, mean, cov, carbon_threshold=1000000, tree_density=False):
    """
    Computing the distribution of carbon for a specified
    site. We assume that the carbon is distributed normally
    around the gps location.
    Inputs:
        site (str) : The site for which we compute the
            carbon distribution
        img_shape (tuple) : The size of the image
            corresponding to the site, this will
            determine the size of the carbon
            distribution matrix
        trees (pd.DataFrame) : The data frame containing
            the relevant parts of the field data
        cov (np.array) : The covariance we assume for the
            normal distribution
    Returns:
        (np.array) 2D array representing the carbon value
            at each pixel of the RGB image
    """
    # calculate carbon distributions with gaussian distribution
    sigma_multiple = 10
    
    max_x_tree = int(sigma_multiple * np.sqrt(cov[0,0]))
    max_y_tree = int(sigma_multiple * np.sqrt(cov[1,1]))
    y_range, x_range = np.mgrid[0:max_y_tree, 0:max_x_tree]
    pos = np.dstack((y_range, x_range))

    #TODO investigate mean in normal distribution
    rv = multivariate_normal([max_y_tree/2 + mean[1], max_x_tree/2 + + mean[0]], np.flip(cov))
    
    gaussian = rv.pdf(pos)
    gaussian = rv.pdf(pos)
    trees_site = trees[trees.site == site]
    max_x = img_shape[2]
    max_y = img_shape[1]
    carbon_distribution = np.zeros((img_shape[1], img_shape[2]))
    for _, tree in trees_site[trees_site.carbon < carbon_threshold].iterrows():
        gaussian_tree = gaussian * tree.carbon
        if tree_density:
            gaussian_tree = gaussian * np.mean(trees_site[trees_site.carbon < carbon_threshold].carbon.values)
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

    '''
    #normalizing distribution so that total carbon mass agrees with field data
    total_carbon_site = trees_site["carbon"].sum()
    total_carbon_distribution = carbon_distribution.sum()
    carbon_distribution = carbon_distribution * total_carbon_site / total_carbon_distribution
    '''

    if carbon_distribution.sum() != trees_site.loc[trees_site.carbon < carbon_threshold, "carbon"].sum():
        offset = abs(carbon_distribution.sum() - trees_site.loc[trees_site.carbon < carbon_threshold, "carbon"].sum())
        logging.warning("Total carbon in distribution is off by %s", offset)

    return carbon_distribution
