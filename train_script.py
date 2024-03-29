import json

from transformer_ee.train import NCtrainer

with open("transformer_ee/config/input.json", encoding="UTF-8", mode="r") as f:
    input_d = json.load(f)
# input_d["data_path"]="transformer_ee/data/NCsample_without_MVA_selection.csv.xz"
input_d["model"]["name"] = "Transformer_EE_v4"
# input_d["model"]["kwargs"]["nhead"] = 12
input_d["model"]["epochs"] = 20
input_d["model"]["kwargs"]["num_layers"] = 5
input_d["optimizer"]["name"] = "sgd"
input_d["optimizer"]["kwargs"]["lr"] = 0.01
input_d["optimizer"]["kwargs"]["momentum"] = 0.9
input_d["save_path"] = "save/model/NC/weights"
# input_d["weight"] = {"name": "FlatSpectraWeights", "kwargs": {"maxweight": 5, "minweight": 0.2}}
my_trainer = NCtrainer(input_d)
my_trainer.train()
my_trainer.eval()
