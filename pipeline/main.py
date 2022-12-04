import pandas as pd
import numpy as np
import logging
import rasterio
from parts.patches import pad
from parts.boundary import create_boundary
from parts.estimate_carbon import compute_carbon_distribution
from parts.helper.constants import get_gps_error
from parts.processing import process
from parts.data_split import train_val_test_dataloader, compute_mean
import argparse
from PIL import Image
import os
import json
# from IPython import embed
from parts.model import SimpleCNN, Resnet18Benchmark, train
import torch
import torch.nn as nn
from torchvision.transforms import Normalize, Resize
from parts.benchmark_dataset import create_benchmark_dataset, train_val_test_dataloader_benchmark

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

    # creating folder "paths["dataset"]/processed" if it doesn't exist
    if not os.path.exists(paths["dataset"] + "sites"):
        logging.info("Creating directory %s", paths["dataset"] + "sites")
        os.makedirs(paths["dataset"] + "sites")

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

        unpadded_img= img
        img = pad(img, patch_size) #padding image to make patches even
        carbon_distribution = compute_carbon_distribution(site, img.shape, trees, mean, covariance)
        assert img.shape[1:] == carbon_distribution.shape


        np.save(paths["dataset"] + "sites/" + f"{site}_carbon", carbon_distribution)
        im = Image.fromarray(np.moveaxis(img, 0, -1)[:,:,:3])
        im_unpadded = Image.fromarray(np.moveaxis(unpadded_img, 0, -1)[:, :, :3])
        im.save(paths["dataset"] + "sites/" + f"{site}_image.png")
        im_unpadded.save(paths["dataset"] + "sites/" + f"{site}_unpadded_image.png")


def main():
    path_to_main = os.path.dirname(__file__) 
    logging.basicConfig(filename=path_to_main + "/pipeline.log", level=logging.INFO, 
            filemode="w", format="[%(asctime)s | %(levelname)s] %(message)s")

    # #TODO REMINDER: comment this section out to run the code without an argparser
    # args = get_args()
    # create_dataset= args.createpatches
    # process_dataset=args.processpatches
    # splits=args.splitting
    # batch_size= args.batchsize

    #TODO REMINDER: Uncomment this section to change the following hyperparameters without using an argparser
    create_dataset= False
    process_dataset= False
    splits=[4,1,1]
    batch_size= 16

    # hyperparameters
    #TODO REMINDER: Run it with create_dataset=True and 28*2*2*2*2 patch_size
    # then change create_dataset=False and change patch_size to any integer you want dividing 28*2*2*2*2 (to avoid creating the dataset again)
    # particularly this allow for patch_size 224 (required for pretrained resnets)

    with open(f"{path_to_main}/config.json", "r") as cfg:
        hyperparameters = json.load(cfg)

    #Setting paths to local scratch when running on cluster
    if hyperparameters["cluster"]:
        logging.info("Changing paths to compute node local scratch: %s", os.environ.get("TMPDIR"))
        paths = {
                "dataset" : os.environ.get("TMPDIR") + "/",
                "reforestree" : os.environ.get("TMPDIR") + "/reforestree/"
        }
    else:
        paths = {
            "dataset": hyperparameters["dataset"], 
            "reforestree" : hyperparameters["reforestree"]
        }

    trees = pd.read_csv(paths["reforestree"] + "/field_data.csv")
    trees = trees[["site", "X", "Y", "lat", "lon", "carbon"]]

    if create_dataset:
        logging.info("Creating data")
        create_data(paths, hyperparameters, trees)

    if process_dataset:
        data = process(trees.site.unique(), hyperparameters, paths)
    else:
        data= pd.read_csv(paths["dataset"]+"patches_df.csv", usecols=["carbon", "path", "site", "rotation", "patch size", "site_index"])

    data = data.reset_index()

    logging.info("Dataset has %s elements", len(data))
    
    transform = None

    #Computing mean and std of pixels and normailzing accordingly
    if hyperparameters["normalize"]:
        logging.info("Normalizing data")
        mean, std = compute_mean(hyperparameters, data, paths["dataset"])
        transform = Normalize(mean, std) 


    train_loader, val_loader, test_loader= train_val_test_dataloader(paths["dataset"], data, splits=splits,
                                                                     batch_size=batch_size, transform=transform)

    #To check that the dataloader works as intended
    # batch_example= next(iter(train_loader))

    #Training hyperparameters
    training_hyperparameters = {
        "learning_rate" : 1e-6,
        "n_epochs" : 10,
        "loss_fn": nn.MSELoss(),
        "log_interval": 1,
        "device": "cpu",
        #TODO: Add mode options when it come to optimizer expect the Adam and AMSGrad
        "optimizer": "amsgrad"
    }

    if torch.cuda.is_available():
        logging.info("Using cuda")
        training_hyperparameters["device"]="cuda"
    elif torch.has_mps:
        logging.info("Using mps")
        training_hyperparameters["device"]="mps"

    #TODO: Find a good one :)

    #Training a simple model
    #simple_cnn = SimpleCNN(hyperparameters["patch_size"], 3)
    #train(simple_cnn, training_hyperparameters, train_loader)

    # Training a Resnet18 model (patch size needs to be 224 for now as the transforms are not working)
    #resnet_benchmark= Resnet18Benchmark()
    #train(resnet_benchmark, training_hyperparameters, train_loader)

    # Dataloaders for the benchmark dataset
    create_benchmark_dataset(paths) # can comment this out if the benchmark dataset was already created
    benchmark_dataset= pd.read_csv(paths["dataset"]+ "benchmark_dataset.csv")
    train_benchmark, val_benchmark, test_benchmark = train_val_test_dataloader_benchmark(benchmark_dataset, splits=[4, 1, 1],
                                                                                batch_size=32, transform=Resize(224))
    tree_img_sampled, carbon_sample, site_sample = next(iter(train_loader))

main()
