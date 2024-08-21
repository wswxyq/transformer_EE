"""
This module contains the functions to make predictions using the trained model, AKA inference.
It shares the same data loading and preprocessing functions with the training module. 
However, the target values are not provided in the input data and weights are not calculated. See transformer_ee/dataloader/pd_dataset.py for more details.
"""

import os
import json
import pandas as pd
import torch
import numpy as np
from transformer_ee.utils import get_gpu
from transformer_ee.dataloader.pd_dataset import Normalized_pandas_Dataset_with_cache
from .load_model_checkpoint import load_model_checkpoint


class Predictor:
    """
    A class to make predictions using a trained model.
    """

    def __init__(self, model_dir: str, dtframe: pd.DataFrame):
        self.gpu_device = get_gpu()  # get gpu device
        self.model_dir = model_dir
        with open(
            os.path.join(self.model_dir, "input.json"), encoding="UTF-8", mode="r"
        ) as fc:
            self.train_config = json.load(fc)
        print("Loading model...")
        self.net = load_model_checkpoint(self.model_dir)
        self.net.eval()
        self.net.to(self.gpu_device)
        print("Loading dataset...")
        self.dataset = Normalized_pandas_Dataset_with_cache(
            self.train_config,
            dtframe,
            eval=True,
            use_cache=False,
        )
        self.train_stat = {}
        with open(
            os.path.join(self.model_dir, "trainset_stat.json"),
            encoding="UTF-8",
            mode="r",
        ) as fs:
            self.train_stat = json.load(fs)
        self.dataset.normalize(self.train_stat)
        print("Creating dataloader...")
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.train_config["batch_size_test"],
            shuffle=False,
            num_workers=10,
        )

    def go(self):
        """
        A function to make predictions using the trained model.
        """
        prediction = []
        with torch.no_grad():
            for _batch_idx, batch in enumerate(self.dataloader):
                print("Batch: ", _batch_idx + 1, " / ", len(self.dataloader))
                vector_valid_batch = batch[0].to(self.gpu_device)
                scalar_valid_batch = batch[1].to(self.gpu_device)
                mask_valid_batch = batch[2].to(self.gpu_device)
                Netout = self.net.forward(
                    vector_valid_batch, scalar_valid_batch, mask_valid_batch
                )
                prediction.append((Netout.detach().cpu().numpy()))
        prediction = np.concatenate(prediction)
        return prediction
