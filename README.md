# AI4good_ai4est

Automating the estimation of carbon stock for forestry sites is an important yet challenging task due to the inherent diversity of the problem. Recently, the advancements in machine learning algorithms, and especially deep learning techniques, gave rise to previously unexplored solutions. However, in order to tackle this problem using deep learning, it is necessary to combine efforts and collaborate. To that end, the benchmark forestry dataset ReforesTree was created using a collection of diligently hand collected measurements and drone images. In the current work, we further explore ReforesTree and strive to bring light to its shortcomings and limitations. We propose a collection of approaches for estimating carbon stock from images, while tackling the GPS noise, matching discrepancies and outlying samples found in the underlying dataset. Our methods significantly reduces the need for pre-processing procedures that inevitably produce additional noise in the final dataset. Furthermore, they are able to account for the GPS noise and smooth out the carbon distribution. Finally, a comparison between the pipeline found in the original paper and the ones developed in this work is presented. The trained models are unable to make meaningful predictions on unknown data and fail on capturing the underlying carbon distribution.

## Context and Documentation

This project was conducted as part of the course AI4Good in Fall 2022 offered at ETH Zurich. For more details on the course refer to the [course website](https://www.vorlesungen.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?lerneinheitId=163124&semkez=2022W&ansicht=LEHRVERANSTALTUNGEN&lang=en). We uploaded the final project report and presentation to the repository. They summarize our findings and provide additional documentation. We hope they are useful to other researchers. If you have further questions, feel free to reach out.


## Different implemented runs

As described in our report, different approaches were considered throughout the project: the point mass approach, the gaussian approach, the tree
density approach and the benchmark. For each of these runs a different configuration of the main function needs to be used. To simplify the execution,
we created a configuration file containing the appropriate parameters for each of these runs. The next section describes how to execute one
of these runs.


## Executing the code

To execute the code after the repository has been cloned, a virtual environment needs to be created from the requirements file. Additionally
the orginal dataset needs to be downloaded from the following [link](https://zenodo.org/record/6813783) and extracted. Then depending on 
the different runs and machines used to execute the code, the right configuration file needs to be chosen from the `pipeline/configs` directory.

Here's a rundown of the necessary steps to run the code on a local machine

  1. Identify the correct config file
  2. Set `cluster` to `false`in this config file
  2. Set this config file in line 29 of `pipeline/main.py`
  2. Create a dataset directory where the processed dataset will be stored
  3. Change the path to dataset to the directory created in step 2 in the config file
  4. Change the path to reforestree to point to the location of the reforestree dataset on you machine (so /path/to/dir/reforestree/)
  5. Change the path to main to point to where this directory lies on your machine (so /path/to/dir/ai4est/)
  5. Activate environment
  6. Execute code from inside the `pipeline` directory by running `python3 main.py`
  
If the code shall be run on the cluster the necessary steps differ slightly

  1. Identify the correct config file
  2. Set this config file in line 29 of `pipeline/main.py`
  2. Set `cluster` to `true` in the config file chosen in step 1.
  5. Change the path to main to point to where this directory lies on your machine (so /path/to/dir/ai4est/)
  5. Change the path to reforestree to point to where the reforestree directory lies on the cluster (so /path/to/dir/reforestree/)
  3. Change the paths in the `submit.sh` file in the `pipeline` directory to the paths pointing to your  python virtualenv created 
      from the requirements file.
  4. Execute code from inside `pipeline` directory by running `sbatch < submit.sh`
  
## Results

For every experimental run a set of results is produced that can be found inside the results folder. For each one of the files found in the results folder, a constant convention is followed. The name of the file indicates the site used for testing the model when the later was trained on the other 5 sites of the dataset. Please note that some of the files (in particular most csv files) are produced during the run of the pipeline. To have access to all of the files mentioned below (in particular the plots), it is necessary to execute the `plots_report.ipynb` notebook. Simply specify the `run_name` in the second cell of the notebook and execute it. Afterwards the content of the folder can be broken down into the parts presented below.

  1. The csv folder contains the following:
      
      - The losses folder contains six csv files (one for each site) that store the loss progression observed during training. The contents of the csv file consist of three columns, one for the epoch in which the losses were documented, and two for the losses observed. 
    
      - The metrics.csv file contains information relevant to the training and testing of the models for every run. Its information include the testing site for the respective run, the patch size used, the degreed of rotations deployed, the number of patches created, the total carbon that is present in all the patches of the testing site, along with the respective value predicted by the model, the mean carbon value for the testing site, along with the respective mean prediction given by the model, and finally, the MSE, RMSE and R squared metrics for the run.
    
      - The predictions folder contains six csv files (one for each site) that store the testing results. The contents of the csv file consist of two columns, one for the ground truth labels and the other one for the prediction made by the model.
    
      - The results.csv file has 9 columns. The first seven columns contain information relevant to the creation of the respective patch (e.g. carbon, path, site, degrees of rotation, size, patch index inside of the site). The last two columns constitute a summary of the six csv files that can be found in the predictions folder discussed above.
    
  2. The plots folder contains the following:
    
      - The losses folder contains 6 pdf files and 6 png files. The pdf files present the plotting of the loss per epoch for each run. The png files present the exact same plotting while also giving a comparison between the current run and all the previous ones.
    
      - The predictions folder contains three different types of pdf files, 6 for each type. First, predictions_*site name* RGB.pdf gives a visual representation of the ground truth carbon distribution as well as the one predicted by the model. Second, predictions_hist_*site name* RGB.pdf contains the histograms of the target and predicted values for the model. Third, predictions_sorted_*site name* RGB is equivalent to predictions_*site name* RGB.pdf, but instead of plotting the results on the site images, the target values are sorted and compared directly to the model's predictions.

## Code Structure
Here we want to provide an overview of the structure and most important parts of the repository.

```
.
├── data                        
│   ├── annotations             # Our hand annotations for one site 
│   ├── dataset                 # Will be created when running the pipeline 
│   └── reforestree             # Original data to be downloaded as described    
|   └── ...             
├── exploration                 # Jupyter notebooks for data analysis
│   ├── plots_report.ipynb      # Notebook to create plots of run results
│   ├── data_ground_truth.ipynb # Notebook analyzing GPS error, carbon and boundaries
│   ├── analyse_matching.ipynb  # Notebook for initial matching analysis
|   └── ...
├── pipeline                    # Main code
│   ├── configs                 # Predefined config files for experiments described in report
│   ├── parts                   # All parts of the pipeline
│   ├── main.py                 # Execute this file to run the pipeline
|   └── ...
├── results                     # Already contains csv files of our experiments
└── ...
```
