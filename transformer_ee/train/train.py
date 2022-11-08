"""
A module for training a model.
"""

import os

import torch

from transformer_ee.dataloader import Pandas_NC_Dataset, split_data
from transformer_ee.utils import get_gpu, hash_dict
from transformer_ee.model import create_model
from .optimizer import create_optimizer
from .loss import get_loss_function


class NCtrainer:
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
        self.dataset = Pandas_NC_Dataset(self.input_d)
        self.trainloader, self.validloader, self.testloader = split_data(
            self.dataset, self.input_d
        )
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

        self.epochs = input_d.get("epochs", 100)
        self.bestscore = 1e9
        self.print_interval = input_d.get("print_interval", 1)

        self.save_path = os.path.join(
            input_d.get("save_path", "."), "model" + hash_dict(self.input_d)
        )
        os.makedirs(
            self.save_path,
            exist_ok=True,
        )

    def train(self):
        r"""
        Train the model.
        """
        for i in range(self.epochs):

            self.net.train()  # begin training

            batch_train_loss = []
            
            for (batch_idx, batch) in enumerate(self.trainloader):

                vector_train_batch = batch[0].to(self.gpu_device)
                scalar_train_batch = batch[1].to(self.gpu_device)
                mask_train_batch = batch[2].to(self.gpu_device)
                target_train_batch = batch[3].to(self.gpu_device)

                Netout = self.net.forward(
                    vector_train_batch, scalar_train_batch, mask_train_batch
                )
                # This will call the forward function, usually it returns tensors.

                loss = get_loss_function(
                    self.input_d["loss"]["name"], Netout, target_train_batch
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

            self.net.to("cpu")

            batch_valid_loss = []

            for (batch_idx, batch) in enumerate(self.validloader):
                vector_valid_batch = batch[0]
                scalar_valid_batch = batch[1]
                mask_valid_batch = batch[2]
                target_valid_batch = batch[3]

                Netout = self.net.forward(
                    vector_valid_batch, scalar_valid_batch, mask_valid_batch
                )
                # This will call the forward function, usually it returns tensors.

                loss = torch.mean(
                    torch.abs((Netout - target_valid_batch) / target_valid_batch)
                )
                batch_valid_loss.append(loss)
                if batch_idx % self.print_interval == 0:
                    print(
                        "Epoch: {}, batch: {} Loss: {:0.4f}".format(i, batch_idx, loss)
                    )
            self.valid_loss_list_per_epoch.append(
                torch.mean(torch.Tensor(batch_valid_loss))
            )

            self.net.to(self.gpu_device)


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
                    os.path.join(self.save_path, "best_model.zip")
                )
                print("model saved with best score: {:0.4f}".format(self.bestscore))
