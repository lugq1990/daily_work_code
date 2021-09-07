# -*- coding:utf-8 -*-
"""This is to implement the linear regression with PyTorch.
Linear regression is just with: y = w * x + b.
As I write the code, I use SGD as optimizer with learning rate with 1,
I get nan loss, how is that possible?
This is caused by exploding vanish problem: http://neuralnetworksanddeeplearning.com/chap5.html
so I just choose more advance optimizer with Adam, in fact, with Adam we don't need to set
learning rate as it is a self-adapt solution, but I also change the default learning rate
I find that it's more efficient!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from matplotlib import style
import tempfile
import os
import shutil

style.use('ggplot')

# hyper-parameters
input_size = 13
output_size = 1
epochs = 300
learning_rate = .01


class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        # as features is 13D also we need to keep the bias
        # the output is just 1D with regression prediction

    def forward(self, x):
        x = x.float()
        x = self.fc(x)
        return x


# here I just load boston data from sklearn for linear regression
x, y = load_boston(return_X_y=True)
y = y.reshape(-1, 1)

# we should split data to train and test
train_position = int(len(x) * .9)
xtrain = x[:train_position]
ytrain = y[:train_position]
xtest = x[train_position:]
ytest = y[train_position:]

# we should convert data to Tensor
xtrain = torch.from_numpy(xtrain)
xtest = torch.from_numpy(xtest)

# have to convert the target to float type
ytrain = torch.from_numpy(ytrain).view(-1, 1).float()
ytest = torch.from_numpy(ytest).view(-1, 1).float()

# after load data, we should create our model
model = LinearNet()
print("Model Structure: ", model)

# we should create the optimizer and loss function
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

train_loss_list = []
test_loss_list = []

# start training epochs
for epoch in range(epochs):
    # get the prediction on train data
    out = model(xtrain)

    # get the loss based on train data
    loss = criterion(out, ytrain)

    # backward step to get the gradient
    optimizer.zero_grad()
    loss.backward()

    # update the model parameters
    optimizer.step()

    # make the test loss based on updated parameters
    # we don't need the gradient for test data
    with torch.no_grad():
        out_test = model(xtest)
        loss_test = criterion(out_test, ytest)

    train_loss_list.append(loss)
    test_loss_list.append(loss_test)

    if epoch % 10 == 0:
        print("[Epoch: %d, train loss: %.2f, test loss: %.2f]" %
              (epoch, loss.item(), loss_test.item()))

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
