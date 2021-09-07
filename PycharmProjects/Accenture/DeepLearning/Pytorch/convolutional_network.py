# -*- coding:utf-8 -*-
"""This is to implement CNN model structure using PyTorch.
As CNN is used for image, so if you are given some images,
like to do classification like is this a cat or dog?
Then what you need to do is to use CNN model to extract
information from image. For CNN there are two important parts:
1. local perception 2. weights sharing. That's means that we could
use kernel to extract information, most times with kernel size
with square like: 1*1, 3*3, 5*5; then you could use this kernel
to slide on the image, with dot product with each pixel and kernel,
for weights sharing, means during the kernel sliding on the image,
weights doesn't change, so that we do reduce the parameters in CNN!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib import style
import os, shutil, tempfile

style.use('ggplot')

# we could even place the tensor to GPU if there is at least one GPU available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# hyper-parameters
n_classes = 10
batch_size = 100    # there will be a trade-off between batch_size and learning_rate
epochs = 5
learning_rate = .01

# here we could use MNIST data for example
train_set = torchvision.datasets.MNIST(root='./data', train=True,
                                       transform=transforms.ToTensor(), download=True)
test_set = torchvision.datasets.MNIST(root='./data', train=False,
                                      transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                          batch_size=batch_size, shuffle=False)


# In fact, we could make our CNN class two ways: one is as the general way; two
# we could use sequence module to combine the whole model layers, just like "Keras"
class CNNNetG(nn.Module):
    def __init__(self):
        super(CNNNetG, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.batch1 = nn.BatchNorm2d(16)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.batch2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(7 * 7 * 32, n_classes)

    def forward(self, x):
        x = x.float()
        x = self.activation(self.batch1(self.conv1(x)))
        x = self.pool(x)
        x = self.activation(self.batch2(self.conv2(x)))
        x = self.pool(x)
        x = x.reshape(x.size()[0], -1)    # flatten the data
        x = self.fc(x)
        return x


class CNNNetS(nn.Module):
    """
    We could also use Sequence to combine different layers into one object,
    that you could see that we do find that when call layers, really simple!
    """
    def __init__(self):
        super(CNNNetS, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(7 * 7 * 32, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.size()[0], -1)    # we should flatten the result, first dimension is batch_size, others are features
        x = self.fc(x)
        return x


# instant our model, we could also put our model to GPU if available
model_g = CNNNetG()
model_g = model_g.to(device)
print("General model structure: ", model_g)

model_s = CNNNetS()
model_s = model_s.to(device)
print("Sequence model strcture: ", model_s)

# to test two ways, here create another new object to represent the model
model = model_s.to(device)

# instant optimizer and loss evaluation
optimizer = optim.Adam(model_s.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

train_loss_list = []
test_acc_list = []

# oohu, start training :)
n_train = len(train_loader)
for epoch in range(epochs):
    curr_train_loss = []
    for i, (images, labels) in enumerate(train_loader):
        # we could put images and labels to GPU if available
        # as I remember that PyTorch will "copy" the image to GPU!
        images = images.to(device)
        labels = labels.to(device)
        # we will get batch_size images per-epoch
        # get prediction
        pred_images = model(images)

        # get loss
        loss_train = criterion(pred_images, labels)

        # backward step
        optimizer.zero_grad()
        loss_train.backward()

        # update model parameters
        optimizer.step()

        curr_train_loss.append(loss_train)

    # store each epoch train loss
    train_loss_list.append(curr_train_loss)

    # each epoch will evaluate test data
    # as we add batch-normalization, so here we could eval model for test to let
    # model use whole data mean/variance not based on mini-batch mean/variance
    model_s.eval()
    with torch.no_grad():
        total_size = 0
        correct_pred = 0
        for images, labels in test_loader:
            # we have to put test data also to GPU if available, otherwise will raise error
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, pred = torch.max(outputs.data, 1)   # get the prediction result
            total_size += labels.size()[0]
            correct_pred += (pred == labels).sum().item()

        test_acc_list.append((correct_pred / total_size) * 100)
    print("{Epoch: %d: test accuracy: %.2f}" % (epoch, test_acc_list[epoch]))

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

