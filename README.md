# AI4good_re4estree


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
  
 
 
