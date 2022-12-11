import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_losses(run_name, site, losses):
    plt.plot(losses.epoch, losses.train_loss)
    plt.plot(losses.epoch, losses.val_loss)
    plt.title(f'Training Results for {site} site')
    plt.xlabel('Epoch(s)')
    plt.ylabel('Loss')
    endrange = (np.max(losses.epoch) // 5) + 1
    plt.xticks(range(0, endrange * 5, 5))
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f'results/{run_name}/plots/losses/train_val_results_{site}.png')

def create_predictions_csv(dataset_path, run_name):
    # DO NOT CHANGE THE ORDER. Thanks <3
    sites = ['Nestor Macias RGB',
             'Leonor Aspiazu RGB',
             'Carlos Vera Arteaga RGB',
             'Carlos Vera Guevara RGB',
             'Flora Pluas RGB',
             'Manuel Macias RGB']

    final_csv = pd.read_csv(dataset_path + "patches_df.csv")
    predictions = []
    target = []

    csv_path = f'results/{run_name}/csv/'
    for site in sites:
        df = pd.read_csv(csv_path + f'predictions/predictions_{site}.csv')
        predictions.append(df['preds'].values.tolist())
        target.append(df['true_value'].values.tolist())
    predictions = [item for sublist in predictions for item in sublist]
    target = [item for sublist in target for item in sublist]

    final_csv['predictions'] = predictions
    final_csv['true_value'] = target
    final_csv.to_csv(csv_path + 'predictions.csv')
    return final_csv

def report_results(dataset_path, run_name):
    predictions = create_predictions_csv(dataset_path, run_name)

if __name__ == "__main__":
    paths = {
        "dataset" : "/Users/timengelmann/GitHub/ai4est/data/dataset/",
        "reforestree" : "/Users/timengelmann/GitHub/ai4est/data/reforestree/"
    }
    
    report_results(paths['dataset'], 'baseline')
