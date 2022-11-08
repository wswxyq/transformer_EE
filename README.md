# transformer_EE
A transformer encoder based neutrino energy estimator.

## Prerequisites

It is recommended to use a container to run the code. 

* Python 3.10 was used for development.
* PyTorch 1.13 was used for development. Pytorch 1.13 supports both CUDA and Apple MPS. By default, the code will use CUDA if it is available. Otherwise, it will use Apple MPS.
* Pandas
* Numpy

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

