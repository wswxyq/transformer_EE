# %%
import json
from transformer_ee.train import NCtrainer

with open("transformer_ee/config/input.json", encoding="UTF-8", mode="r") as f:
    input_d = json.load(f)
# %%
my_trainer = NCtrainer(input_d)
my_trainer.train()

# %%
