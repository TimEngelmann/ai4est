import pandas as pd


def report_results(dataset_path):
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

    csv_path = './testing_results/'
    for site in sites:
        df = pd.read_csv(csv_path + f'{site}.csv')
        predictions.append(df['preds'].values.tolist())
        target.append(df['true_value'].values.tolist())
    predictions = [item for sublist in predictions for item in sublist]
    target = [item for sublist in target for item in sublist]

    final_csv['predictions'] = predictions
    final_csv['true_value'] = target
    final_csv.to_csv('predictions.csv')
