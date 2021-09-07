# -*- coding:utf-8 -*-
"""This is to implement feed forward network based on PyTorch.
For feed forward network, here I just want to show maybe there
are many layers with dense connection with each other.
As you could implement as many layers you want, but one thing
should remember is vanishing/ exploding gradient problem!
Easy to understand: if you have 5 layers and each layer gradient
is 10, then with 5 layers' pass, then the gradient will be 10**5
as we will do multiply with each layer, that's vanishing gradient;
suppose you just get gradient with 0.1, then when you pass gradient
to first layer, the gradient is just 0.1**5=0.00001, so small! so
you couldn't learn anything! There you many other way to solve this:
Relu, BatchNormalization, RNN, CNN, etc. I will make it deeper in future
tutorials.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tempfile
import shutil
import os

style.use('ggplot')

# hyper-parameters
input_size = 4
output_size = 3
hidden_units = 128
learning_rate = .01
epochs = 300


# define feed forward network
class FeedForwardNet(nn.Module):
    def __init__(self):
        super(FeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, output_size)

    def forward(self, x):
        # fix this error: add x.float()
        # RuntimeError: Expected object of scalar type Float but got scalar type Double for argument #4 'mat1'
        x = x.float()
        x = F.relu(F.dropout(self.fc1(x)))   # also add dropout to prevent overfiting
        x = F.relu(self.fc2(x))
        x = F.softmax(input=x, dim=1)   # make sum to 1 with row use softmax
        return x


x, y = load_iris(return_X_y=True)

# split data to train and test data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.2, random_state=1234)

# convert ndarray to Tensor
xtrain = torch.from_numpy(xtrain)
xtest = torch.from_numpy(xtest)

# as for multi-class, pytorch need label type to be long()!
# for binary, to change type to float()
ytrain = torch.from_numpy(ytrain).long().squeeze_()
ytest = torch.from_numpy(ytest).long().squeeze_()

# instant our model
model = FeedForwardNet()
print("Model Structure: ", model)

# instant optimizer and loss object
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

train_loss_list = []
test_loss_list = []

# start training, oohu
for epoch in range(epochs):
    # get train prediction
    pred_train = model(xtrain)

    # get loss
    loss_train = criterion(pred_train, ytrain)

    # backward step
    optimizer.zero_grad()
    loss_train.backward()

    # get the gradient of model parameters
    optimizer.step()

    # get test loss
    # we don't need gradient for test data
    with torch.no_grad():
        pred_test = model(xtest)
        loss_test = criterion(pred_test, ytest)

    train_loss_list.append(loss_train)
    test_loss_list.append(loss_test)

    if epoch % 10 == 0:
        print("{Epoch: %d, train loss: %.2f, test loss: %.2f}" % (epoch, loss_train, loss_test))

print("Whole training step finished!")

# then we could plot the train and test data loss curve
plt.plot(range(len(train_loss_list)), train_loss_list, label='training_loss')
plt.plot(range(len(test_loss_list)), test_loss_list, label='testing_loss')

plt.title("Training and testing loss curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.show()

# we could save our trained model checkpoint to disk, here I just save
# it to a temperate folder, you could just change the path you want
tmp_path = tempfile.mkdtemp()
torch.save(model.state_dict(), os.path.join(tmp_path, 'model.ckpt'))
print("Get model:", os.listdir(tmp_path))
# remove the temperate folder
shutil.rmtree(tmp_path)