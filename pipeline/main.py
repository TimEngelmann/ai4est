import pandas as pd
import numpy as np
import logging
from parts.patches import pad
from parts.boundary import create_boundary
from parts.estimate_carbon import compute_carbon_distribution
from parts.helper.constants import get_gps_error
from parts.helper.argumentparser import get_args
from parts.processing import process
from parts.data_split import train_val_test_dataloader, compute_mean
from PIL import Image
import os
import json
from IPython import embed
from parts.model import SimpleCNN, Resnet18Benchmark, train, test
import torch
import torch.nn as nn
from torchvision.transforms import Normalize, Resize, Compose
from parts.benchmark_dataset import create_benchmark_dataset, train_val_test_dataloader_benchmark
from parts.helper.datacreation import create_data
from parts.evaluation import report_results, plot_losses


def main():
    path_to_main = os.path.dirname(__file__) 
    logging.basicConfig(filename=path_to_main + "/pipeline.log", level=logging.INFO, 
            filemode="w", format="[%(asctime)s | %(levelname)s] %(message)s")

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

    if hyperparameters["create_dataset"]:
        logging.info("Creating data")
        create_data(paths, hyperparameters, trees)
    if hyperparameters["process_dataset"]:
        data = process(trees.site.unique(), hyperparameters, paths)
    if hyperparameters["benchmark_dataset"]:
        # benchmark_dataset
        data = create_benchmark_dataset(paths)
    else:
        data= pd.read_csv(paths["dataset"]+"patches_df.csv", usecols=["carbon", "path", "site", "rotation", "patch size", "site_index"])

    logging.info("Dataset has %s elements", len(data))
    transform = None

    #Computing mean and std of pixels and normalizing accordingly
    if hyperparameters["normalize"]:
        logging.info("Normalizing data")
        mean, std = compute_mean(hyperparameters, data, paths["dataset"], hyperparameters["benchmark_dataset"])
        logging.info("Computed - Mean: {} and Std: {}".format(mean, std))
        transform = Compose([
            Normalize(mean, std),
            Resize((224, 224))
        ])

    #Training hyperparameters
    training_hyperparameters = {
        "learning_rate" : 1e-5,
        "n_epochs" : 30,
        "loss_fn": nn.MSELoss(),
        "log_interval": 20,
        "device": "cpu",
        "optimizer": "amsgrad",
        "batch_size": 64
    }

    if torch.cuda.is_available():
        logging.info("Using cuda")
        training_hyperparameters["device"]="cuda"
    elif torch.has_mps:
        logging.info("Using mps")
        training_hyperparameters["device"]="mps"

    # create directories
    for directory in [
        f'results/{hyperparameters["run_name"]}/plots/losses',
        f'results/{hyperparameters["run_name"]}/csv/losses',
        f'results/{hyperparameters["run_name"]}/plots/predictions',
        f'results/{hyperparameters["run_name"]}/csv/predictions']:
        if not os.path.exists(directory):
                os.makedirs(directory)
    
    sites = data.site.unique()

    # Training a Resnet18 model (patch size needs to be 224 for now as the transforms are not working)
    splits = {"training":[], "validation":[], "testing":[]}
    for i in range(len(sites)):
        splits["testing"] = [sites[i]]
        splits["validation"] = [sites[i]]
        splits["training"] = list(np.delete(sites, i))
        logging.info("Training on sites {}".format(splits["training"]))
        logging.info("Validating on site number {}".format(splits["validation"]))
        logging.info("Testing on site number {}".format(splits["testing"]))
       
        if hyperparameters["benchmark_dataset"]:
            train_loader, val_loader, test_loader = train_val_test_dataloader_benchmark(data, splits=splits,
                                                                                        batch_size=training_hyperparameters["batch_size"], transform=transform)
        else:
            train_loader, val_loader, test_loader = train_val_test_dataloader(paths["dataset"], data, splits=splits,
                                                                            batch_size=training_hyperparameters["batch_size"], transform=transform)

        site_name = splits["testing"][0]
        resnet_benchmark = Resnet18Benchmark()
        model, losses = train(resnet_benchmark, training_hyperparameters, train_loader, val_loader, site_name)
        
        logging.info(f'Saving Losses')
        losses.to_csv(f'results/{hyperparameters["run_name"]}/csv/losses/losses_{site_name}.csv')
        plot_losses(hyperparameters["run_name"], site_name, losses)

        logging.info(f'Testing Model')
        if test_loader is None:
            test_loader = val_loader
        test_results = test(model, test_loader, training_hyperparameters["loss_fn"], training_hyperparameters["device"])
        test_results.to_csv(f'results/{hyperparameters["run_name"]}/csv/predictions/predictions_{site_name}.csv')

    if not hyperparameters["benchmark_dataset"]:
        report_results(paths["dataset"], hyperparameters["run_name"])

main()
