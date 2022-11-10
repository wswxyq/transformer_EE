import json
from transformer_ee.train import NCtrainer

with open("transformer_ee/config/input_CC.json", encoding="UTF-8", mode="r") as f:
    input_d = json.load(f)
input_d["model"]["name"] = "Transformer_EE_v4"
input_d["model"]["epochs"] = 500
my_trainer = NCtrainer(input_d)
my_trainer.train()
my_trainer.eval()
