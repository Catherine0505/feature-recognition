import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import skimage.io as skio
import skimage
from skimage.transform import resize
import scipy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from load_nosetip_data import training_dataloader, validation_dataloader

class NosetipNet(nn.Module):
    """
    Nosetip detection neural network.
    """
    def __init__(self):
        super(NosetipNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding = [1, 1])
        self.norm1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 5, padding =  [2, 2])
        self.norm2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 12, 5, padding = [2, 2])
        self.norm3 = nn.BatchNorm2d(12)
        # self.conv4 = nn.Conv2d(26, 32, 7, padding = [3, 3])
        # self.norm4 = nn.BatchNorm2d(32)
        # size: batch_size * (32, 3, 4)
        self.fc1 = nn.Linear(360, 27)
        self.norm5 = nn.BatchNorm1d(27)
        self.fc2 = nn.Linear(27, 2)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.norm1(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, (2, 2))

        x = self.conv2(x)
        # x = self.norm2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, (2, 2))

        x = self.conv3(x)
        # x = self.norm3(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, (3, 3))

        # x = self.conv4(x)
        # x = self.norm4(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, (3, 3))

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.norm5(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


training_data, training_set = training_dataloader(24)
validation_data, validation_set = validation_dataloader()

for i, batch in enumerate(training_set):
    height, width = batch["image"][0].numpy().shape
    print(type(batch))
    for j in range(7, 9):
        plt.imshow(batch["image"].numpy()[j] + 0.5, cmap = "gray")
        plt.plot(batch["feature"].numpy()[j][1] * width,
            batch["feature"].numpy()[j][0] * height, linestyle = "none",
            marker = ".", markersize = 12, color = 'g')
        plt.show()
    break

# Set hyperparameters.
model = NosetipNet()
optimizer = optim.Adam(model.parameters(), lr = 4e-3)
criterion = nn.MSELoss()
num_epoch = 100
training_losses = []
validation_losses = []

# Start training for 100 epochs.
for iteration in range(num_epoch):
    training_loss_it = []
    validation_loss_it = []
    for i, batch in enumerate(training_set):
        x = batch["image"].unsqueeze(1)
        y = batch["feature"]
        output = model(x)
        training_loss = criterion(y, output).float()
        # if i == 0:
        #     print(output)
        #     print(y)
        #     print(((output - y) ** 2).sum() / y.numpy().shape[0])
        #     print(loss)
        training_loss_it.append(training_loss.item() * x.shape[0])
        optimizer.zero_grad()
        training_loss.backward()
        optimizer.step()
    training_losses.append(np.sum(training_loss_it) / len(training_data))

    model.eval()
    for i, batch in enumerate(validation_set):
        validation_x = batch["image"].unsqueeze(1)
        validation_y = batch["feature"]
        validation_output = model(validation_x)
        validation_loss = criterion(validation_y, validation_output).float()
        validation_loss_it.append(validation_loss.item() * validation_x.shape[0])
    validation_losses.append(np.sum(validation_loss_it) / len(validation_data))
    model.train()

# Plot the training and validation loss curve.
plt.plot(list(range(num_epoch)), np.log(training_losses), label = "Training loss")
plt.plot(list(range(num_epoch)), np.log(validation_losses),
    label = "Validation loss")
plt.legend()
plt.show()

# Display training results.
for i, batch in enumerate(training_set):
    training_x = batch["image"].unsqueeze(1)
    training_output = model(training_x)
    height, width = batch["image"][0].numpy().shape
    for j in range(7, 13):
        plt.imshow(batch["image"][j].numpy() + 0.5, cmap = "gray")
        plt.plot(batch["feature"][j].numpy()[1] * width,
            batch["feature"][j].numpy()[0] * height, linestyle = "none",
            marker = ".", markersize = 12, color = 'g')
        plt.plot(training_output[j].detach().numpy()[1] * width,
            training_output[j].detach().numpy()[0] * height, linestyle = "none",
            marker = ".", markersize = 12, color = 'r')
        plt.show()
    break

# Display validation results. 
for i, batch in enumerate(validation_set):
    validation_x = batch["image"].unsqueeze(1)
    validation_output = model(validation_x)
    print(validation_output.detach().numpy())
    height, width = batch["image"][0].numpy().shape
    for j in range(7, 13):
        plt.imshow(batch["image"][j].numpy() + 0.5, cmap = "gray")
        plt.plot(batch["feature"][j].numpy()[1] * width,
            batch["feature"][j].numpy()[0] * height, linestyle = "none",
            marker = ".", markersize = 12, color = 'g')
        plt.plot(validation_output[j].detach().numpy()[1] * width,
            validation_output[j].detach().numpy()[0] * height, linestyle = "none",
            marker = ".", markersize = 12, color = 'r')
        plt.show()
    break
