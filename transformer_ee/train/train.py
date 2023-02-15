"""
A module for training a model.
"""

import os
import json

import torch
import numpy as np
from matplotlib import pyplot as plt

from transformer_ee.dataloader import Pandas_NC_Dataset, split_data
from transformer_ee.utils import get_gpu, hash_dict
from transformer_ee.model import create_model
from transformer_ee.utils import plot_xstat, plot_y_hist, plot_2d_hist_count
from .optimizer import create_optimizer
from .loss import get_loss_function
from .loss_track import plot_loss


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
        print(json.dumps(self.input_d, indent=4))
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

        self.epochs = input_d["model"].get("epochs", 100)
        print("epochs: ", self.epochs)
        self.bestscore = 1e9
        self.print_interval = input_d.get("print_interval", 1)

        self.save_path = os.path.join(
            input_d.get("save_path", "."), "model_" + hash_dict(self.input_d)
        )
        os.makedirs(
            self.save_path,
            exist_ok=True,
        )
        _json_name = os.path.join(self.save_path, "input.json")
        if not os.path.exists(_json_name):
            with open(_json_name, "w") as f:
                json.dump(self.input_d, f, indent=4)

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
                weight_train_batch = batch[4].to(self.gpu_device)[
                    :, None
                ]  # add a dimension

                Netout = self.net.forward(
                    vector_train_batch, scalar_train_batch, mask_train_batch
                )
                # This will call the forward function, usually it returns tensors.

                loss = get_loss_function(
                    self.input_d["loss"]["name"],
                    Netout,
                    target_train_batch,
                    weight=weight_train_batch,
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

            for (batch_idx, batch) in enumerate(self.validloader):
                vector_valid_batch = batch[0].to(self.gpu_device)
                scalar_valid_batch = batch[1].to(self.gpu_device)
                mask_valid_batch = batch[2].to(self.gpu_device)
                target_valid_batch = batch[3].to(self.gpu_device)
                weight_valid_batch = batch[4].to(self.gpu_device)[:, None]

                Netout = self.net.forward(
                    vector_valid_batch, scalar_valid_batch, mask_valid_batch
                )
                # This will call the forward function, usually it returns tensors.

                loss = get_loss_function(
                    self.input_d["loss"]["name"],
                    Netout,
                    target_valid_batch,
                    weight=weight_valid_batch,
                )

                batch_valid_loss.append(loss)
                if batch_idx % self.print_interval == 0:
                    print(
                        "Epoch: {}, batch: {} Loss: {:0.4f}".format(i, batch_idx, loss)
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

            plot_loss(
                self.train_loss_list_per_epoch,
                self.valid_loss_list_per_epoch,
                self.save_path,
            )

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

        trueval = []
        prediction = []

        self.net.cpu()
        self.net.to(self.gpu_device)

        for (_batch_idx, batch) in enumerate(self.testloader):
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

        truet = trueval[:, 0]
        predt = prediction[:, 0]

        resolution = (predt - truet) / truet

        with open(os.path.join(self.save_path, "result.txt"), "w") as f:
            f.write("mean resolution: {:0.4f} \n".format(np.mean(resolution)))
            f.write("std resolution: {:0.4f} \n".format(np.std(resolution)))
            f.write(
                "rms resolution: {:0.4f} \n".format(np.sqrt(np.mean(resolution**2)))
            )

        print("mean resolution: ", np.mean(resolution))
        print("std resolution: ", np.std(resolution))
        print("rms resolution: ", np.sqrt(np.mean(resolution**2)))

        filename = "testset_result"
        basename = filename
        _, _, _ = plot_xstat(
            truet,
            (predt - truet) / truet,
            bins=50,
            range=(0, 5),
            outdir=self.save_path,
            name=basename + "_xstat",
            title=filename,
            ext="png",
            xlabel="True E",
            ylabel="(Reco E - True E) / True E",
        )
        _, _, _ = plot_xstat(
            predt,
            (predt - truet) / truet,
            bins=50,
            range=(0, 5),
            outdir=self.save_path,
            name=basename + "_xstat_xreco",
            title=filename,
            ext="png",
            xlabel="Reco E",
            ylabel="(Reco E - True E) / True E",
        )

        plot_y_hist(
            (predt - truet) / truet,
            range=(-1, 1),
            outdir=self.save_path,
            name=basename + "_yhist",
            title=filename,
            bins=200,
            ext="png",
            xlabel="(Reco E - True E) / True E",
        )

        plot_2d_hist_count(
            truet,
            predt,
            outdir=self.save_path,
            name=basename + "_2dhist",
            ext="png",
            xlabel="True E",
            ylabel="Reco E",
        )
