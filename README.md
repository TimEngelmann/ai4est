# AI4good_re4estree

Automating the estimation of carbon stock for forestry sites is an important yet challenging task due to the inherent diversity of the problem. Recently, the advancements in machine learning algorithms, and especially deep learning techniques, gave rise to previously unexplored solutions. However, in order to tackle this problem using deep learning, it is necessary to combine efforts and collaborate. With that aim in mind, the benchmark forestry dataset ReforesTree was created using a collection of diligently hand collected measurements and drone images. In the current work, we further explore ReforesTree and strive to bring light to its shortcomings and limitations. We propose a collection of approaches for estimating carbon stock from drone images, while tackling the GPS noise, matching discrepancies and outlying samples found in the original ReforesTree dataset.

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
  2. Set this config file in line 29 of `pipeline/main.py`
  2. Create a dataset directory where the processed dataset will be stored
  3. Change the path to dataset to the directory created in step 2 in the config file
  4. Change the path to reforestree to point to the location of the reforestree dataset on you machine
  5. Activate environment
  6. Execute code from inside the `pipeline` directory by running `python3 main.py`
  
If the code shall be run on the cluster the necessary steps differ slightly

  1. Identify the correct config file
  2. Set this config file in line 29 of `pipeline/main.py`
  2. Set `cluster` to `true` in the config file chosen in step 1.
  3. Change the paths in the `submit.sh` file in the `pipeline` directory to the
      paths pointing to your reforestree folder and the python virtualenv created from 
      the requirements file.
  4. Execute code from inside `pipeline` directory by running `sbatch < submit.sh`
  
## Results

For every experimental run a set of results is produced that can be found inside the results folder and can be broken down into as presented below:

  1. The csv folder contains the following:
    
    a) the losses folder contains six csv files (one for each site) that store the loss progression observed when a model is trained on 5 sites and tested on the left out site, which shares the same name with the csv file. The contents of the csv file consist of three columns, one for the epoch in which the losses were documented, and two for the losses observed. 
    
    b) the metrics.csv file contains information relevant to the training and testing of the models for every run. Its information include the testing site for the respective run, the patch size used, the degreed of rotations deployed, the number of patches created, the total carbon that is present in all the patches of the testing site, along with the respective value predicted by the model, the mean carbon value for the testing site, along with the respective mean prediction given by the model, and finally, the MSE, RMSE and R squared metrics for the run.
    
    c) the predictions folder contains six csv files (one for each site) that store the results produced when a model is trained on 5 sites and tested on the left out site, which shares the same name with the csv file. The contents of the csv file consist of two columns, one for the ground truth labels and the other one for the prediction made by the model.
    
    d) the results.csv file has 9 columns. The first seven columns contain information relevant to the creation of the respective patch (e.g. carbon, path, site, degrees of rotation, size, patch index inside of the site). The last two columns constitute a summary of the six csvs that can be found in the predictions folder previously discussed.
    
  2. The plots folder contains the following:
    
    a) the losses folder 6 pdf files and 6 png files. The pdf files present the plotting of the loss per epoch for each run. The files share the same name with the site that is being tested for each particular run. The png files present the exact same plotting while also giving a comparison between the current run and all the previous ones.
    
    b) the predictions folder contains three different types of pdf files, 6 for each type. First, predictions_*site name* RGB.pdf gives a visual representation of the ground truth carbon distribution as well as the predicted one, when the model is tested on the respective site. Second, predictions_hist_*site name* RGB.pdf contains the histograms of the target and predicted values for the model when tested on the *site name* site. Third, predictions_sorted_*site name* RGB is equivalent to predictions_*site name* RGB.pdf, but instead of plotting the results on the site images, this time the target values are sorted and compared directly to the model's predictions.
