# -*- coding:utf-8 -*-
"""
This is to test with word cloud that take some text as input
and output with a figure for most frequent words with different
size for different importance.

@author: Guangqiang.lu
"""
import os
import matplotlib.pyplot as plt
import tempfile
import shutil
from wordcloud import WordCloud

# sample text
text = """
In the above examples, we had to manually implement both the forward and backward passes of our neural network. Manually implementing the backward pass is not a big deal for a small two-layer network, but can quickly get very hairy for large complex networks.

Thankfully, we can use automatic differentiation to automate the computation of backward passes in neural networks. The autograd package in PyTorch provides exactly this functionality. When using autograd, the forward pass of your network will define a computational graph; nodes in the graph will be Tensors, and edges will be functions that produce output Tensors from input Tensors. Backpropagating through this graph then allows you to easily compute gradients.

This sounds complicated, it’s pretty simple to use in practice. Each Tensor represents a node in a computational graph. If x is a Tensor that has x.requires_grad=True then x.grad is another Tensor holding the gradient of x with respect to some scalar value.

Here we use PyTorch Tensors and autograd to implement our two-layer network; now we no longer need to manually implement the backward pass through the network:
"""

# tmp_path = tempfile.mkdtemp()
# with open(os.path.join(tmp_path, 'test.txt'), 'w') as f:
#     f.write(text)


def plot_diff_freq_diff_size(text=None, file_name=None):
    # if file_name is not None:
    #     with open(os.path.join(tmp_path, file_name), 'r') as f:
    #         text = f.read()

    if text is None:
        raise ValueError("Please provide with either text or file_name to show!")

    wordcloud = WordCloud().generate(text)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    plot_diff_freq_diff_size(text)

