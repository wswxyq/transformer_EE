import json
import gc # garbage collector

from transformer_ee.train import NCtrainer

for i in [128, 256, 512, 1024, 4096, 8192]:
    with open("transformer_ee/config/input.json", encoding="UTF-8", mode="r") as f:
        input_d = json.load(f)
    input_d["model"]["name"] = "Transformer_EE_v4"
    input_d["model"]["epochs"] = 400
    #input_d["model"]["kwargs"]["nhead"] = 8
    input_d["model"]["kwargs"]["dim_feedforward"] = i
    input_d["optimizer"]["name"] = "sgd"
    input_d["optimizer"]["kwargs"]["lr"] = 1e-2
    input_d["optimizer"]["kwargs"]["momentum"] = 0.9
    input_d["save_path"] = "save/model/NC/group"
    #input_d["loss"]["name"] = "mean absolute percentage error squared"
    my_trainer = NCtrainer(input_d)
    my_trainer.train()
    my_trainer.eval()
    del my_trainer
    gc.collect()

