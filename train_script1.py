import json
from transformer_ee.train import NCtrainer

with open("transformer_ee/config/input.json", encoding="UTF-8", mode="r") as f:
    input_d = json.load(f)

input_d["model"]["name"]="Transformer_EE_v2"
input_d["model"]["kwargs"] = {
    "num_layers": 2,
    "dim_feedforward": 1024,
    "dropout": 0.1,
}
my_trainer = NCtrainer(input_d)
my_trainer.train()
my_trainer.eval()
