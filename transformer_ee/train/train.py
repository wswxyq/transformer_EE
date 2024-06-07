"""
A module for training a model.
"""

import json
import os
import logging

import numpy as np
import torch

from transformer_ee.dataloader.load import get_train_valid_test_dataloader
from transformer_ee.model import create_model
from transformer_ee.utils import (
    get_gpu,
    hash_dict,
)

from .loss import linear_combination_loss
from .loss_track import plot_loss
from .optimizer import create_optimizer


class MVtrainer:
    r"""
    A class to train a model.

    You may want to load config from a json file. For example:
    ```

    with open("transformer_ee/config/input.json", encoding="UTF-8", mode="r") as f:
        input_d = json.load(f)

    ```
    """

    def __init__(self, input_d: dict):
        self.gpu_device = get_gpu()  # get gpu device
        self.input_d = input_d
        print(json.dumps(self.input_d, indent=4))

        (
            self.trainloader,
            self.validloader,
            self.testloader,
            self.train_set_stat,
        ) = get_train_valid_test_dataloader(input_d)
        self.net = create_model(self.input_d).to(self.gpu_device)
        self.optimizer = create_optimizer(self.input_d, self.net)
        self.bestscore = 1e9
        self.print_interval = self.input_d.get("print_interval", 1)

        self.train_loss_list_per_epoch = (
            []
        )  # list to store training losses after each epoch
        self.valid_loss_list_per_epoch = (
            []
        )  # list to store validation losses after each epoch

        self.epochs = input_d["model"].get("epochs", 100)
        print("epochs: ", self.epochs)
        self.bestscore = 1e9
        self.print_interval = input_d.get("print_interval", 1)

        self.save_path = os.path.join(
            #input_d.get("save_path", "."), "model_" + hash_dict(self.input_d)
            input_d.get("save_path", "."), "model_" + input_d.get("model_phys_name")
        )
        print("save_path: ",self.save_path)
        os.makedirs(
            self.save_path,
            exist_ok=True,
        )
        input_json = os.path.join(self.save_path, "input.json")
        if not os.path.exists(input_json):
            with open(input_json, "w") as f:
                json.dump(self.input_d, f, indent=4)

        stat_json = os.path.join(self.save_path, "trainset_stat.json")
        if not os.path.exists(stat_json):
            with open(stat_json, "w") as f:
                json.dump(self.train_set_stat, f, indent=4)

        # Set up logger
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger()      

        # Create a file handler
        fh = logging.FileHandler(self.save_path + "/training.log")
        fh.setLevel(logging.INFO)

          # Create a console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create a formatter and set it for both handlers
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
  

    def train(self):
        r"""
        Train the model.
        """
        for i in range(self.epochs):
            self.net.train()  # begin training

            batch_train_loss = []

            for batch_idx, batch in enumerate(self.trainloader):

                vector_train_batch = batch[0].to(
                    self.gpu_device
                )  # shape: (batch_size, max_seq_len, vector_dim)
                scalar_train_batch = batch[1].to(
                    self.gpu_device
                )  # shape: (batch_size, scalar_dim)
                mask_train_batch = batch[2].to(
                    self.gpu_device
                )  # shape: (batch_size, max_seq_len)
                target_train_batch = batch[3].to(
                    self.gpu_device
                )  # shape: (batch_size, target_dim)
                weight_train_batch = batch[4].to(
                    self.gpu_device
                )  # shape: (batch_size, 1)
                Netout = self.net.forward(
                    vector_train_batch, scalar_train_batch, mask_train_batch
                )
                # This will call the forward function, usually it returns tensors.

                loss = linear_combination_loss(
                    Netout,
                    target_train_batch,
                    weight=weight_train_batch,
                    **self.input_d["loss"]["kwargs"],
                )  # regression loss

                # Zero the gradients before running the backward pass.
                self.optimizer.zero_grad()

                # Backward pass: compute gradient of the loss with respect to all the learnable
                # parameters of the model. Internally, the parameters of each Module are stored
                # in Tensors with requires_grad=True, so this call will compute gradients for
                # all learnable parameters in the model.
                loss.backward()
                # Calling the step function on an Optimizer makes an update to its
                # parameters
                self.optimizer.step()

                batch_train_loss.append(loss)
                if batch_idx % self.print_interval == 0:
                    print(
                        "Epoch: {}, batch: {} Loss: {:0.4f}".format(i, batch_idx, loss)
                    )
            self.train_loss_list_per_epoch.append(
                torch.mean(torch.Tensor(batch_train_loss))
            )

            self.net.eval()  # begin validation

            # self.net.to("cpu")

            batch_valid_loss = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(self.validloader):
                    vector_valid_batch = batch[0].to(self.gpu_device)
                    scalar_valid_batch = batch[1].to(self.gpu_device)
                    mask_valid_batch = batch[2].to(self.gpu_device)
                    target_valid_batch = batch[3].to(self.gpu_device)
                    weight_valid_batch = batch[4].to(self.gpu_device)

                    Netout = self.net.forward(
                        vector_valid_batch, scalar_valid_batch, mask_valid_batch
                    )
                    # This will call the forward function, usually it returns tensors.

                    loss = linear_combination_loss(
                        Netout,
                        target_valid_batch,
                        weight=weight_valid_batch,
                        **self.input_d["loss"]["kwargs"],
                    )

                    batch_valid_loss.append(loss)
                    if batch_idx % self.print_interval == 0:
                        print(
                            "Epoch: {}, batch: {} Loss: {:0.4f}".format(
                                i, batch_idx, loss
                            )
                        )
                self.valid_loss_list_per_epoch.append(
                    torch.mean(torch.Tensor(batch_valid_loss))
                )

            # self.net.to(self.gpu_device)

            print(
                "Epoch: {}, train_loss: {:0.4f}, valid_loss: {:0.4f}".format(
                    i,
                    self.train_loss_list_per_epoch[-1],
                    self.valid_loss_list_per_epoch[-1],
                )
            )

            if self.valid_loss_list_per_epoch[-1] < self.bestscore:
                self.bestscore = self.valid_loss_list_per_epoch[-1]
                torch.save(
                    self.net.state_dict(),
                    os.path.join(self.save_path, "best_model.zip"),
                )
                print("model saved with best score: {:0.4f}".format(self.bestscore))
                print("model saved at: ",self.save_path)

            plot_loss(
                self.train_loss_list_per_epoch,
                self.valid_loss_list_per_epoch,
                self.save_path,
            )

            self.logger.info(f"Epoch [{i}] - Training Loss,  Validation Loss: {self.train_loss_list_per_epoch[-1]:.4f}, {self.valid_loss_list_per_epoch[-1]:.4f}")
    

    def eval(self):
        r"""
        Evaluate the model.
        """
        self.net.load_state_dict(
            torch.load(
                os.path.join(self.save_path, "best_model.zip"),
                map_location=torch.device("cpu"),
            )
        )
        self.net.eval()
        self.net.to(self.gpu_device)

        trueval = []
        prediction = []

        for _batch_idx, batch in enumerate(self.testloader):
            vector_valid_batch = batch[0].to(self.gpu_device)
            scalar_valid_batch = batch[1].to(self.gpu_device)
            mask_valid_batch = batch[2].to(self.gpu_device)
            target_valid_batch = batch[3].to(self.gpu_device)

            Netout = self.net.forward(
                vector_valid_batch, scalar_valid_batch, mask_valid_batch
            )
            trueval.append(target_valid_batch.detach().cpu().numpy())
            prediction.append((Netout.detach().cpu().numpy()))

        trueval = np.concatenate(trueval)  # [:,0]
        prediction = np.concatenate(prediction)
        np.savez(
            os.path.join(self.save_path, "result.npz"),
            trueval=trueval,
            prediction=prediction,
        )
