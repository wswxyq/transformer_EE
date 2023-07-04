# transformer_EE
A transformer encoder based neutrino energy estimator. This is a highly flexible frame work which allows for easy modification of the model, loss function, and data loader. Currently, the code supports CSV file with array stored as a comma separated string. We provide example scripts for NOvA and DUNE experiments.

## Prerequisites

It is recommended to use a venv/container to run the code. 

* Python 3.10 was used for development.
* PyTorch 1.13 was used for development. Pytorch 1.13 supports both CUDA and Apple MPS. By default, the code will use CUDA if it is available. Otherwise, it will use Apple MPS.
* Pandas
* Numpy
* Matplotlib >= 3.5

## Set up
PYTHONPATH should be set to the top level directory of the repository.

For example, if this repo is cloned to /home/user/transformer_EE, then the following should be added to your .bashrc file:
```
export PYTHONPATH=/home/user/transformer_EE:$PYTHONPATH
```

Test that the PYTHONPATH is set correctly by running the following command:
```
echo $PYTHONPATH
```
The commandline should return:
```
/home/user/transformer_EE:
```

## Running the code
There is an example script for the NC dataset. To run the code, run the following command:
```
python3 script/train_script.py
```

## Configuring the code
The config file is a json file. The default config file is located at transformer_ee/config.
There are two ways to configure the code:
1. Edit the config file directly.
2. Modify the dictionary in the script/train_script.py file. For example, to select the model, add the following line:
```
input_d["model"]["name"] = "Transformer_EE_v4"
```
in the script/train_script.py file.