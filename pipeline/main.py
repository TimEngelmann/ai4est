import pandas as pd
import numpy as np
import logging
import rasterio
from parts.patches import pad
from parts.boundary import create_boundary
from parts.estimate_carbon import compute_carbon_distribution
from parts.helper.constants import get_gps_error
from parts.processing import process
from parts.data_split import train_val_test_dataloader
import argparse
from PIL import Image
import os


#argument parser
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--createpatches", required=False, type=str2bool,
        default="True",help="boolean for whether to create the patches images dataset")
    ap.add_argument("-b", "--batchsize", required=False, type=int,
                    default=64, help="batch size for dataloader")
    ap.add_argument("-s", "--splitting", nargs='+', required=False, type=float,
                    default=[4,1,1], help="list of length 3 [size_train, size_val, size_test]. "
                                        "If summing up to 1, the data will be split randomly across each site,"
                                        "if summing up to 6, the data data will be split by site")
    return ap.parse_args()




def create_data(paths, hyperparameters, trees):
    """
    Combining RGB image data and carbon distribution into
    4-channel image for each site.
    """

    patch_size = hyperparameters["patch_size"]

    for site in trees.site.unique():
        logging.info("Creating data for site %s", site)

        #get covariance for normal distribution
        if isinstance(hyperparameters["covariance"], dict):
            covariance = np.array(hyperparameters["covariance"][site])
        else:
            covariance = np.array(hyperparameters["covariance"])
        mean = np.array(hyperparameters["mean"])

        boundary = create_boundary(site, paths["reforestree"])
        img_path = paths["reforestree"] + f"wwf_ecuador/RGB Orthomosaics/{site}.tif"

        #masking the drone image using the boundary
        with rasterio.open(img_path) as raster:
            img, _ = rasterio.mask.mask(raster, boundary, crop=False)

        img = pad(img, patch_size) #padding image to make patches even
        carbon_distribution = compute_carbon_distribution(site, img.shape, trees, mean, covariance)
        assert img.shape[1:] == carbon_distribution.shape

        # creating folder "paths["dataset"]/processed" if it doesn't exist
        if not os.path.exists(paths["dataset"] + "sites"):
            logging.info("Creating directory %s", paths["dataset"] + "sites")
            os.makedirs(paths["dataset"] + "sites")

        np.save(paths["dataset"] + "sites/" + f"{site}_carbon", carbon_distribution)
        im = Image.fromarray(np.moveaxis(img, 0, -1))
        im.save(paths["dataset"] + "sites/" + f"{site}_image.png")    




def main():
    logging.basicConfig(filename="pipeline.log", level=logging.INFO, 
            filemode="w", format="[%(asctime)s | %(levelname)s] %(message)s")

    #TODO: comment this section out to run the code without an argparser
    args = get_args()
    create_dataset= args.createpatches
    splits=args.splitting
    batch_size= args.batchsize

    # hyperparameters
    hyperparameters = {
        "patch_size" : 400,
        "filter_white" : True,
        "angle" : 30,
        "rotations" : [0, 30, 60],
        "mean": [-246.35193671, 57.03964288],
        "covariance" : [[106196.72698492, -24666.11304593], [-24666.11304593, 113349.22307974]]
    }

    #import gps error
    gps_error = get_gps_error()

    #TODO: Change path names to your own local directories
    paths = {
        "reforestree" : "/cluster/work/igp_psr/ai4good/group-3b/reforestree/",
        "dataset" : "/cluster/work/igp_psr/ai4good/group-3b/data/"
    }

    '''
    paths = {
        "reforestree" : "data/reforestree/",
        "dataset" : "data/dataset/"
    }
    '''

    trees = pd.read_csv(paths["reforestree"] + "field_data.csv")
    trees = trees[["site", "X", "Y", "lat", "lon", "carbon"]]
    if create_dataset:
        logging.info("Creating data")
        create_data(paths, hyperparameters, trees)

    data = process(trees.site.unique(), hyperparameters, paths)

    train_loader, val_loader, test_loader= train_val_test_dataloader(paths["dataset"], data,
                                                                 splits=splits, batch_size=batch_size)                                                          

main()
