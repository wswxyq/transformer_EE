# %%
import torch
import pandas as pd
import json
import transformer_ee.dataloader.string_conv as string_conv
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset


# %%
mps_device = torch.device("mps")

# %%
df = pd.read_csv(
    "transformer_ee/data/2022-08-21_rnne_NC_250_fGScatter_20MeV_KE_01e8_max_trackE_cut.csv.xz"
)

with open("transformer_ee/config/input.json", "r") as f:
    input_d = json.load(f)


# %%
for particle_feature in input_d["vector"]:
    df[particle_feature] = df[particle_feature].apply(string_conv.string_to_float_list)


# %%
class Pandas_NC_Dataset(Dataset):
    def __init__(self, dataframe, config: dict):
        self.df = dataframe
        self.len = len(dataframe)
        self.maxpronglen = config["max_num_prongs"]
        self.vectornames = config["vector"]
        self.scalarnames = config["scalar"]
        self.targetname = config["target"]

        # calculate mean and std for normalization
        self.stat_scalar = []
        for x in self.scalarnames:
            self.stat_scalar.append([df[x].mean(), df[x].std()])
        self.stat_scalar = torch.Tensor(self.stat_scalar).T
        self.stat_scalar = self.stat_scalar[:, None, :]

        self.stat_vector = []
        for x in self.vectornames:
            _tmp = []
            for y in df[x]:
                _tmp.extend(y)
            self.stat_vector.append([np.mean(_tmp), np.std(_tmp)])
        self.stat_vector = torch.Tensor(self.stat_vector).T
        self.stat_vector = self.stat_vector[:, None, :]
        self.d={}

    def __getitem__(self, index):
        if index in self.d:
            return self.d[index]

        row = self.df.iloc[index]
        _vectorsize = len(row[self.vectornames[0]])
        _vector = torch.Tensor(row[self.vectornames]).T
        _scalar = torch.Tensor(row[self.scalarnames]).T
        _vector = (_vector - self.stat_vector[0]) / self.stat_vector[1]
        _scalar = (_scalar - self.stat_scalar[0]) / self.stat_scalar[1]

        return_tuple = (
            # pad the vector to maxpronglen with zeros
            F.pad(_vector, (0, 0, 0, self.maxpronglen - _vectorsize), "constant", 0),
            # return the scalar
            _scalar,
            # return src_key_padding_mask
            F.pad(
                torch.zeros(_vectorsize, dtype=torch.bool),
                (0, self.maxpronglen - _vectorsize),
                "constant",
                1, # pad with True
            ),
            torch.Tensor(row[self.targetname]),
        )
        self.d[index] = return_tuple
        return return_tuple

    def __len__(self):
        return self.len


# %%
dataset = Pandas_NC_Dataset(df, input_d)

batch_size_train = 256
batch_size_test = 1024


# %%
seed = 0
_indices = np.arange(len(df))
np.random.seed(seed)
np.random.shuffle(_indices)
test_size = 0.2
train_indicies = _indices[: int(len(_indices) * (1 - test_size))]
test_indicies = _indices[int(len(_indices) * (1 - test_size)) :]

# %%
from torch.utils.data import Subset

train_dataset = Subset(dataset, train_indicies)
test_dataset = Subset(dataset, test_indicies)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train,
                                            shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test,
                                            shuffle=False)

# %%
from transformer_ee.model.transformerEncoder import Transformer_EE_v1
net=Transformer_EE_v1().to(mps_device)

# %%
import torch.optim as optim
optimizer = optim.Adam(net.parameters(), lr=0.001)

# %%
loss_list = []
epochs = 100
error_list = []
lossfunc = nn.MSELoss()

with torch.profiler.profile(
    schedule=torch.profiler.schedule(
        wait=2,
        warmup=2,
        active=6,
        repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("logs"),
    with_stack=True
) as profiler:

    for i in range(epochs):

        net.train()  # begin training

        for (batch_idx, batch) in enumerate(trainloader):
            vector_train_batch = batch[0].to(mps_device)
            scalar_train_batch = batch[1].to(mps_device)
            mask_train_batch = batch[2].to(mps_device)
            target_train_batch = batch[3].to(mps_device)

            Netout = net.forward(vector_train_batch, mask_train_batch)
            # This will call the forward function, usually it returns tensors.

            loss = lossfunc(Netout, target_train_batch)  # regression loss

            # Zero the gradients before running the backward pass.
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to all the learnable
            # parameters of the model. Internally, the parameters of each Module are stored
            # in Tensors with requires_grad=True, so this call will compute gradients for
            # all learnable parameters in the model.
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()

            loss_list.append(loss)
            if batch_idx % 1 == 0:
                # print("Epoch: {}, batch: {} Loss: {} label_loss:{}".format(i, batch_idx, loss, label_loss_))
                print("Epoch: {}, batch: {} Loss: {:0.4f}".format(i, batch_idx, loss))


# %%
torch.save(net.state_dict(), "transformer_ee/model/transformer_ee_v1.pt")


# %%
from transformer_ee.model.transformerEncoder import Transformer_EE_v1

net=Transformer_EE_v1()
net.load_state_dict(torch.load("transformer_ee/model/transformer_ee_v1.pt"))
net.eval()

# %%
net.eval()
net.to(mps_device)
a, b, c, d=next(iter(testloader))
net.cpu()
e=net.forward(a, c)

from matplotlib import pyplot as plt
trueval=d[:, 0].cpu().detach().numpy()
prediction=e[:, 0].cpu().detach().numpy()
resolution=(prediction-trueval)/trueval
print("mean resolution: ", np.mean(resolution))
print("std resolution: ", np.std(resolution))
plt.hist(resolution, bins=100)
plt.show()
net.to(mps_device)

# %%
plt.scatter(trueval, prediction)
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.xlabel("true value")
plt.ylabel("prediction")
plt.show()

# %%
plt.hist2d(trueval, prediction, bins=(50, 50), cmap=plt.cm.jet, range=[[0, 5], [0, 5]])
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.xlabel("true value")
plt.ylabel("prediction")
plt.colorbar()
plt.show()

# %%
print(net.linear.weight)

# %%
print(net.linear.bias)

# %%
prediction

# %%
a

# %%
plt.scatter(np.sum(np.array(a[:, :, 0], dtype=bool), axis=1), prediction)

# %%
plt.scatter(np.sum(np.array(a[:, :, 0], dtype=bool), axis=1), trueval)

# %%
a[:, :, 0].shape

# %%
np.sum(np.array(a[:, :, 0], dtype=bool), axis=1)

# %%
df['mc.nuE'].hist(bins=100)

# %%



