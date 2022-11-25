### Guide on using the local scratch memory on Euler

1. Request temporary memory when submitting job, in slurm you can do this by creating a submit.sh script which looks as such
 
   ```
   #!/bin/bash

   #SBATCH --tmp=xxxG
   #SBATCH --OTHEROPTIONS=XXX

   commands
   ```
   
   where you add the sbatch options as in the above format. The `--tmp` option the tells slurm to create
   a temporary directory with xxx GB of space.

2. Copy the files necessary for training to the new TMPDIR using the following command 
    ```
    rsync -aq path/to/files/* ${TMPDIR}/path_on_local_scratch/.
    ```
    With the pipeline setup we have now, we only to copy over the content of the `sites` folder and the `field_data.csv` file

3. Change paths in your script to point to the TMPDIR, you can use the `os` python package (preinstalled in the standard library)
    and get the path to the TMPDIR with the command `os.environ.get("TMPDIR")`. Beware that the path returned by that command does not
    end with a slash so that you need to type 
    ```
    os.environ.get("TMPDIR") + "/path_on_local_scratch/filename"
    ```
    if you want to access any of the files copied in step 2. Important: make sure that the processed dataset will be stored in the TMPDIR,
    otherwise the whole setup is pointless.
    
In total our submit script should look something like this

  ```
  #!/bin/bash

  #SBATCH --tmp=100G
  #SBATCH --nodes=1
  #SBATCH --mem-per-cpu=12G
  #SBATCH --gpus=1
  #SBATCH --time=4:00:00


  #activate environnement
  mkdir ${TMPDIR}/path_to_data_on_scratch #creating necessary subdirectories in TMPDIR
  rsync -aq /path_to_data/* ${TMPDIR}/path_to_data_on_scratch/.
  python3 main.py
  rsync -aq ${TMPDIR}/path_to_results_scratch/* /path_to_results/.
  ```
  
Where the last step is only needed if you save files in your TMPDIR, since this directory will be deleted after
the training run, and you need to replace `#activate environement` by the command which activates your environement.
