
# Extended Malicious Packages Hunter -- PyPI

## Deploy
***

### # For GitHub User

Before deploying EMPHunter, please make sure you have Pyhton3 (Python 3.10 is recommended).

You can use ``git clone`` command to copy our project to local.
``cd`` to the root directory of EMPHunter and setting the environment with ``pip install -r requirements.txt``.

The data can be download on: https://zenodo.org/records/15470290

Please replace the following DataShare directory with this DataShare directory:

PyPI-Data/DataShare     ==>     /EMPHunter/PyPI/EMPHunter-PyPI/DataShare

## How to use
***
EMPHunter use config.ini to config. Usually, we first train the FastText-PVDM document embedding model (CCEM) to embedding the code representations into vectors later in the detecting process.
However, if you want to used your own model, you need to set the path to your model and skip the training process. Then you can detect malicious packages. 

In main.py:
You need to modify line 63 to change the directory location of the target source code list.

In config.ini:
You need to modify the following directory names:

line 37 pkg_name: The directory name of the source code.

line 38 proj_name: The directory name for temporary files generated during analysis.

line 39 proj_merged: The directory name for the results.

For the meanings of other parameters, please refer to README.md. If you encounter errors about missing files, you can create empty files first.

The analysis results are located in <proj_merged>_result4.txt. An example of the results is as follows:

The second line contains the absolute path of the target file, which can be used to extract the software package name.


### basic
The [basic] part of configuration set up the most important config for EMPHunter. It is mandatory. 

| para name | default                         | meaning                                                                                                                                                                                         |
| --------- |---------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| vector_size | 300                             | vector_size is used to set the vector size of document vectors embedded from our code reprisentations.                                                                                          |
| window | 5                               | The size of window in FastText-PVDM embedding process.                                                                                                                                          |
| min_count | 1                               | The Minimum API numbers of code representation.                                                                                                                                                 |
|epoch | 200                             | The training epoch of FastText-PVDM.                                                                                                                                                            |  
|workers | 16                              | The number of workers of FastText-PVDM training process.                                                                                                                                        |
|prefix_len | 2                               | For an API in python scripts, they can represented as "a.b.c.x". This para control the max length of API representations. e.g., when prefix_len = 3, a API will be expressed as "b.c.x" format. |
|bad_api_dir | PreProcess/removed_api_list.txt | Some API are commenly used in python project, which can greatly affect the result of code clutering. Appanding the bad APIs in defualt list or setting the path to you own list.   |
|WE_model_name | -                               | The APIEM model name.|
|WE_proj_name | -                               | The APIEM model training data file name in DataShare. Usually, we set the same name as WE_model_name.|
|model_name | -                               | The CCEM model name.|
|action | train                           | "train" for model training process;"detect" for malicious detection. |

### training
The [train] part of configuration is needed when action is "train".

| para name | default    | meaning                                                                                           | 
| --------- |------------|---------------------------------------------------------------------------------------------------|
|proj_name | model_name | The CCEM model training data file name in DataShare. Usually, we set the same name as model_name. |
|dir0 | -          | Inital code representation data file name.                                                        | 
|dir1 | -          | Processing code representation data file name.                                                    | 
|dir2 | -          | Final code representation data file name.                                                         | 
|file_search_mode | all_files  | all_files  or setup_only                                                                          |
|graph_mode | cfg+cg     | Setting "cfg+cg" for canonical code representation and "cg" for uncanonical code representaion.   |
|task_num | 16         | The number of workers when generating code representations.                                       | 

### detection
The [detect] part of configuration is needed when action is "detect"

| para name | default | meaning                                                                       |
| --------- |----|-------------------------------------------------------------------------------|
|pkg_name | Unpackaged_Packages_0609_1_pypi_tar | Path to folder of pkgs waiting to be tested.                                  | 
|proj_name | final-4_0609_1_cfg_setup_c.x | The name of data file in DataShare.                                           | 
|dir0 | -          | Inital code representation data file name.                                    | 
|dir1 | -          | Processing code representation data file name.                                | 
|dir2 | -          | Final code representation data file name.                                     | 
|proj_merged | final-4_0609_1_cfg_setup_c.x_merged | Merged (with benign set) file name.                                           |
|file_search_mode | setup_only | all_files  or setup_only                                                      |
|graph_mode | cfg+cg | cfg+cg   or  cg                                                               |
|rank_mode | p&n | positive or negative or p&n  : The way of taking rank(p-only, n-only and p&n) |
|min_cluster_size | 2 | HDBSCAN's min_cluster_size                                                    | 
|min_samples | 20 | HDBSCAN's min_samples                                                         | 
|task_num | 16 | worker number.                                                                | 
|proj_wait_to_be_merged | final-4_0417_benign_cfg_setup_c.x | name of benign set proj                                                       | 
|negative_proj_name | final-4_0417_malicious_cfg_setup_c.x | name of negative set proj                                                     | 


