# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


def plot_precision_recall_curve(y_true, y_probas,
                                title='Precision-Recall Curve',
                                curves=('micro', 'each_class','positive'), ax=None,
                                figsize=None, cmap='nipy_spectral',
                                title_fontsize="large",
                                text_fontsize="medium"):
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    probas = y_probas

    if 'micro' not in curves and 'each_class' not in curves and 'positive' not in curves:
        raise ValueError('Invalid argument for curves as it '
                         'only takes "micro" or "each_class" or "positive" ')

    # Compute Precision-Recall curve and area for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true, probas[:, i], pos_label=classes[i])

    y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true = np.hstack((1 - y_true, y_true))

    for i in range(len(classes)):
        average_precision[i] = average_precision_score(y_true[:, i],
                                                       probas[:, i])

    # Compute micro-average ROC curve and ROC area
    micro_key = 'micro'
    i = 0
    while micro_key in precision:
        i += 1
        micro_key += str(i)

    precision[micro_key], recall[micro_key], _ = precision_recall_curve(
        y_true.ravel(), probas.ravel())
    average_precision[micro_key] = average_precision_score(y_true, probas,
                                                           average='micro')

    print(average_precision)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    if 'each_class' in curves:
        for i in range(len(classes)):
            color = plt.cm.get_cmap(cmap)(float(i) / len(classes))
            ax.plot(recall[i], precision[i], lw=2,
                    label='Precision-recall curve of class {0} '
                          '(area = {1:0.3f})'.format(classes[i],
                                                     average_precision[i]),
                    color=color)

    if 'micro' in curves:
        ax.plot(recall[micro_key], precision[micro_key],
                label='micro-average Precision-recall curve '
                      '(area = {0:0.3f})'.format(average_precision[micro_key]),
                color='navy', linestyle=':', linewidth=4)

    if 'positive' in curves:
        if len(classes) != 2:
            raise TypeError('Invalidate classes num given. Only two classes is '
                            'supported for positive')
        pos_index = 0
        ax.plot(recall[pos_index], precision[pos_index], lw=2,
                label='Precision-recall curve of positive class'
                      '(area = {1:0.3f})'.format(classes[pos_index],
                                                 average_precision[pos_index]),
                color='navy')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='best', fontsize=text_fontsize)
    return ax

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.svm import SVC
x,y = load_iris(return_X_y=True)
x, y = x[:100,:],y[:100]
x,y = shuffle(x,y)
lr = LogisticRegression(C=.01,penalty='l2')
lr.fit(x,y)
prob =lr.predict_proba(x)
# clf = SVC(C=.001,probability=True)
# clf.fit(x,y)
# prob = clf.predict_proba(x)
plot_precision_recall_curve(y,prob,curves='positive')
plt.show()
