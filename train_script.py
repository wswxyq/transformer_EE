import json

from transformer_ee.train import MVtrainer

with open("transformer_ee/config/input_dune.json", encoding="UTF-8", mode="r") as f:
    input_d = json.load(f)

## Set the path to the input data
# input_d["data_path"]="transformer_ee/data/dataset_lstm_ee_nd_rhc_nonswap_loose_cut.csv.xz"

## Set the number of workers in dataloader. The more workers, the faster the data loading. But it may consume more memory.
input_d["num_workers"] = 10

## Set the model hyperparameters
# input_d["model"]["kwargs"]["nhead"] = 12
input_d["model"]["epochs"] = 20
input_d["model"]["kwargs"]["num_layers"] = 5

## Set the optimizer
input_d["optimizer"]["name"] = "sgd"
input_d["optimizer"]["kwargs"]["lr"] = 0.01
input_d["optimizer"]["kwargs"]["momentum"] = 0.9

## Set the path to save the model
input_d["save_path"] = "save/model/test"

## Set the weighter
# input_d["weight"] = {"name": "FlatSpectraWeights", "kwargs": {"maxweight": 5, "minweight": 0.2}}

input_d["dataframe_type"] = "polars"

## Example of adding noise to the input variables
# input_d["noise"] = {
#     "name": "gaussian",
#     "mean": 0,
#     "std": 0.2,
#     "vector": ["particle.energy", "particle.calE", "particle.nHit"],
#     "scalar": ["event.calE", "event.nHits"],
# }

## Example of using WandBLogger

# from transformer_ee.logger.wandb_train_logger import WandBLogger
# my_logger = WandBLogger(
#     project="test", entity="neutrinoenenergyestimators", config=input_d, dir="save", id="testrun"
# )
# my_trainer = MVtrainer(input_d, logger=my_logger)
my_trainer = MVtrainer(input_d)

my_trainer.train()
my_trainer.eval()
