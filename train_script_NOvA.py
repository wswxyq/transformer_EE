import json

from transformer_ee.train import NCtrainer

with open("transformer_ee/config/input_NOvA.json", encoding="UTF-8", mode="r") as f:
    input_d = json.load(f)

input_d["weight"] = {"name": "FlatSpectraWeights", "kwargs": {"maxweight": 5, "minweight": 0.2}}
input_d["data_path"]="transformer_ee/data/dataset_lstm_ee_fd_fhc_nonswap_loose_cut.csv.xz"

my_trainer = NCtrainer(input_d)
my_trainer.train()
my_trainer.eval()
