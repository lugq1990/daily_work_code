# -*- coding:utf-8 -*-
"""This is according to PyTorch official tutorial to
make nn network with step by step, reference this link:
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    """
    This is the subclass for pytorch nn Module, that we could build network
    more efficient!
    """
    def __init__(self):
        super(Net, self).__init__()
        # 1 means image channel, 6 means output channels, 3 means 3*3 squared kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 68)
        self.fc3 = nn.Linear(68, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.max_feature_dim(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def max_feature_dim(self, x):
        """
        This is just to compute how many dimensions that we have!
        so that we could flatten the image result.
        :param x: layer output
        :return: the number of dimensions
        """
        size = x.size()[1:]
        feature_size = 1
        for s in size:
            feature_size *= s
        return feature_size


# now that we have already create the module class, here we could instance the model
net = Net()
print(net)

# get the model parameters
params = list(net.parameters())
print('Model parameter:', params)
print('get first conv1 shape:', params[0].shape)

# we could define out input data
input_data = torch.randn(1, 1, 32, 32)
target = torch.randn(10)
target = target.view(1, 10)

# we could get the prediction
out = net(input_data)
print(out)

# then we could get the loss, we chould just get the nn module MSE loss
criterion = nn.MSELoss()
loss = criterion(out, target)
print("Get loss: ", loss)


#### HERE Just to get the backward step:
# input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#       -> view -> linear -> relu -> linear -> relu -> linear
#       -> MSELoss
#       -> loss

# now that we have get the output, then we could get the whole model
# back-propagation result for each layer, but here we have to remove
# the zero gradient
# after we have get loss for the model, then we could do backward step
print('*'*20)
print('before bias gradient: ')
net.zero_grad()

# we could get the conv1 bias gradient data before backward step
print(net.conv1.bias.grad)

# after that, we could to through the backward step
# this should based on the scalar output result: loss!
loss.backward()

print('new bias gradient: ')
print(net.conv1.bias.grad)


# as we have get the gradient for each parameters, then we should
# update for the whole model parameters by using python code to update
learning_rate = .01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

print("before loss: ", loss)
print("new loss:", criterion(net(input_data), target))


# or in fact, we could use more advance optimizer for model training
optimizer = optim.SGD(net.parameters(), lr=.01)
out = net(input_data)
loss = criterion(out, target)
# make backward
loss.backward()
optimizer.step()   # update the step
print("with optimizer loss: ", loss)




