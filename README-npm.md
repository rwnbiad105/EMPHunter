
# Extended Malicious Packages Hunter -- npm


## Deploy
***

### # For GitHub User

Before deploying EMPHunter, please make sure you have Pyhton3 (Python 3.10 is recommended).

You can use ``git clone`` command to copy our project to local.
``cd`` to the root directory of EMPHunter and setting the environment with ``pip install -r requirements.txt``.

The data can be download on: https://zenodo.org/records/15470290

Please replace the following DataShare directory with this DataShare directory:

npm-Data/DataShare     ==>     /EMPHunter/npm/EMPHunter-npm/DataShare

## How to use
***
Usually, we first train the FastText-PVDM document embedding model (TEM and SEM) to embedding the code representations into vectors later in the detecting process.

However, if you want to used your own model, you need to set the path to your model and skip the training process. Then you can detect malicious packages. The model 

In main.py:

The model configuration can be found in line 88 to 104. To run a new batch of pkgs, you just need to change line 98 to point to the unpacked target package set and line 104 to name you batch.

In line 88, we_train_batch_name is the TEM name.

In line 101, se_train_batch_name is the SEM name.

In line 129, remove annotation sign if you want to train TEM and SEM.

In line 131, remove annotation sign if you want to process your own benign set.

In line 133, remove annotation sign if you want to process your own malicious set.