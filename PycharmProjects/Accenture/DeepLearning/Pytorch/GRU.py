# -*- coding:utf-8 -*-
"""This is to implement GRU another type of RNN.
As LSTM has really great performance on time series
relation data, but there is also another great RNN
module call GRU, it has lower parameters means it
takes much less to train but could also great performance.
The reason why we don't use pure RNN is that if our data
has longer relation, pure RNN couldn't learn that long relation.
So here comes with Gates based RNN module like LSTM and GRU.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import tempfile
import shutil
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

# we could train the model on GPU if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

# haper-parameters
sequence_size = 28
input_size = 28
hidden_units = 128
n_layers = 2
batch_size = 100
epochs = 5
n_classes = 10
learning_rate = .001

# load MNIST data as training and testing
train_data = torchvision.datasets.MNIST(root='./data', train=True,
                                        transform=transforms.ToTensor(), download=True)
test_data = torchvision.datasets.MNIST(root='./data', train=False,
                                       transform=transforms.ToTensor(), download=True)
trainloader = torch.utils.data.DataLoader(dataset=train_data,
                                          batch_size=batch_size,
                                          shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=test_data,
                                         batch_size=batch_size,
                                         shuffle=False)


# then we could define our GRU model class
class GRUNet(nn.Module):
    def __init__(self):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_units, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_units, n_classes)

    def forward(self, x):
        # RNN has two important parts: hidden states and cell states,
        # here should init our hidden states and cell states with 0
        # this means: how many layers * how many samples * how many hidden units
        h0 = torch.zeros(n_layers, x.size()[0], hidden_units).to(device)

        x = x.float()     # change to float type
        # we could do forward step with GRU
        # pass data and init hidden states and cell states
        # we just want to get outputs without hidden outputs with: batch_size * sequence_size * hidden_size
        outs, _ = self.gru(x, h0)

        # linear output with only the last cell outputs as prediction!
        outs = self.fc(outs[:, -1, :])
        return outs


# as we have define our model, instance the model
model = GRUNet().to(device)
print("Get model structure: ", model)

# the we should init our optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# stop the train loss and test accuracy
train_loss_list = []
test_acc_list = []

# start training step, :)
for e in range(epochs):
    curr_loss = []
    for i, (images, labels) in enumerate(trainloader):
        # reshape our images and to device
        images = images.reshape(-1, sequence_size, input_size).to(device)
        labels = labels.to(device)

        # then we could get prediction
        pred_train = model(images)

        # get our loss based on training prediction and labels
        loss_train = criterion(pred_train, labels)

        # backward step
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        if i % 100 == 0:
            print("{Epoch: %d}: Loss: %.2f" % (e, loss_train))

        # get each batch size loss
        curr_loss.append(loss_train)

    train_loss_list.append(curr_loss)

    # after each epoch, we evaluate our model based on trained model
    with torch.no_grad():
        total = 0
        correct_pred = 0
        for images, labels in testloader:
            images = images.reshape(-1, sequence_size, input_size).to(device)
            labels = labels.to(device)

            # get prediction
            pred_test = model(images)

            _, preds = torch.max(pred_test, 1)
            total += labels.size()[0]
            correct_pred += (preds == labels).sum().item()

        test_acc_list.append((correct_pred / total) * 100)

print("Whole training step finished!")

# Here I just wnat to plot 2 graphs, first is each epochs training loss curve
# the other is the test accuracy curve
fig, ax = plt.subplots(len(train_loss_list), 1, figsize=(12, 18))
for i in range(len(train_loss_list)):
    ax[i].plot(range(len(train_loss_list[i])), train_loss_list[i], label='Epoch_' + str(i + 1))
plt.title("Different epochs training loss")
plt.legend()
plt.show()

fig = plt.figure()
plt.plot(range(len(test_acc_list)), test_acc_list)
plt.title("Test accuracy curve")
plt.xlabel("Epochs")
plt.ylabel("Test accuracy")
plt.grid(True)

plt.show()

# we could save our trained model checkpoint to disk, here I just save
# it to a temperate folder, you could just change the path you want
tmp_path = tempfile.mkdtemp()
torch.save(model.state_dict(), os.path.join(tmp_path, 'model.ckpt'))
print("Get model:", os.listdir(tmp_path))
# remove the temperate folder
shutil.rmtree(tmp_path)

