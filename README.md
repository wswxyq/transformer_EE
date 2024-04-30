# transformer_EE
A transformer encoder based neutrino energy estimator. This is a highly flexible frame work which allows for easy modification of the model, loss function, and data loader. Currently, the code supports CSV file with array stored as a comma separated string. We provide example scripts for NOvA and DUNE experiments.

## New Features

* Transformer encoder based model.
* Customizable loss function (you can design very very complex loss functions).
* Customizable model.
* Customizable optimizer.

## Prerequisites

It is recommended to use a venv/container to run the code.

* Python 3.10 was used for development.
* PyTorch 1.13 was used for development. Pytorch 1.13 supports both CUDA and Apple MPS. By default, the code will use CUDA if it is available. Otherwise, it will use Apple MPS.
* Pandas<=2.0 (newer versions may work but are not tested)
* Numpy
* Matplotlib >= 3.5

### Example conda setup in a Linux system with NVIDIA GPU
1. make sure you have conda installed in your computer. See https://docs.anaconda.com/free/miniconda/miniconda-install/ for more details.
2. create a new conda environment by running `conda create --name new_env_name`
3. activate the new environment by running `conda activate new_env_name`
4. install the required packages by running `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`, see https://pytorch.org/get-started/locally/ for more details.
5. install the other required packages by running `conda install scipy pandas=2.0 numpy matplotlib`

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
input_d["model"]["name"] = "Transformer_EE_MV"
```
in the script/train_script.py file.