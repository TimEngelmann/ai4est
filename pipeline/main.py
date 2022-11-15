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
from IPython import embed
from parts.model import SimpleCNN, Resnet18Benchmark, train
import torch
import torch.nn as nn
import torchvision


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
    ap.add_argument("-c", "--createpatches", required=False, type=str2bool,
        default="False",help="boolean for whether to create the patches images dataset")
    ap.add_argument("-p", "--processpatches", required=False, type=str2bool,
        default="False",help="boolean for whether to process the patches")
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
    process_dataset=args.processpatches
    splits=args.splitting
    batch_size= args.batchsize

    #TODO: Uncomment this section to change the following hyperparameters without using an argparser
    # create_dataset= False
    # process_dataset= False
    # splits=[4,1,1]
    # batch_size= 16

    # hyperparameters
    #TODO: Run it with create_dataset=True and 28*2*2*2*2 patch_size
    # then change create_dataset=False and change patch_size to any integer you want dividing 28*2*2*2*2 (to avoid creating the dataset again)
    # particularly this allow for patch_size 224 (required for pretrained resnets)
    hyperparameters = {
        "patch_size" : 224,
        "filter_white" : True,
        "filter_threshold": 0.8,
        "angle" : 30,
        "rotations" : [0],
        "mean": [-246.35193671, 57.03964288],
        "covariance" : [[106196.72698492, -24666.11304593], [-24666.11304593, 113349.22307974]]
    }

    #import gps error
    gps_error = get_gps_error()

    #TODO: Change path names to your own local directories
    paths = {
        "reforestree" : "/Users/victoriabarenne/ai4good/ReforesTree/",
        "dataset" : "/Users/victoriabarenne/ai4good/dataset/"
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

    if process_dataset:
        data = process(trees.site.unique(), hyperparameters, paths)
    else:
        data= df=pd.read_csv(paths["dataset"]+"patches_df.csv", usecols=["carbon", "path", "site", "rotation", "patch size", "site_index"])


    #TODO: fix Dataloader so that it supports the following transform
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.1307,0,0), (0.3081,0,0))])

    train_loader, val_loader, test_loader= train_val_test_dataloader(paths["dataset"], data,
                                                                 splits=splits, batch_size=batch_size)

    #To check that the dataloader works as intended
    batch_example= next(iter(train_loader))

    #Training hyperparameters
    training_hyperparameters = {
        "learning_rate" : 5 * 1e-4,
        "n_epochs" : 1,
        "loss_fn": nn.MSELoss(),
        "log_interval": 1,
        "device": "cpu",
        "optimizer": "amsgrad" #only available optimizer atm, will implement the possibility for other methods later
    }
    if torch.cuda.is_available():
        training_hyperparameters["device"]="cuda"

    #Training a simple model
    simple_cnn = SimpleCNN(hyperparameters["patch_size"], 3)
    train(simple_cnn, training_hyperparameters, train_loader)

    #Training a Resnet18 model (patch size needs to be 224 for now as the transforms are not working)
    resnet_benchmark= Resnet18Benchmark()
    train(resnet_benchmark, training_hyperparameters, train_loader)


main()
