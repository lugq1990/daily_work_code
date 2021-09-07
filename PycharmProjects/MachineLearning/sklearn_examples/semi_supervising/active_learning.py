# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import load_digits
from sklearn.semi_supervised import label_propagation
from sklearn.metrics import classification_report,confusion_matrix

digits = load_digits()
rng = np.random.RandomState(0)
indices = np.arange(len(digits.data))
rng.shuffle(indices)

x = digits.data[indices[:330]]
y = digits.target[indices[:330]]
images = digits.images[indices[:330]]

n_samples = len(y)
n_labeled_points = 10
max_iterations = 5

unlabeled_indices = np.arange(n_samples)[n_labeled_points:]
f = plt.figure(figsize=(12,10))

for i in range(max_iterations):
    if len(unlabeled_indices) ==0:
        break
    ytrain = np.copy(y)
    ytrain[unlabeled_indices] = -1

    model = label_propagation.LabelPropagation(gamma=.25, max_iter=5)
    #start to train the model
    model.fit(x,ytrain)

    pred = model.transduction_[unlabeled_indices]
    true_labels = y[unlabeled_indices]

    cm = confusion_matrix(true_labels,pred,labels=model.classes_)
    print('iteration {0:d},{1:s}'.format(i,70*"_"))
    print(cm)

    pred_entropies = stats.distributions.entropy(model.label_distributions_.T)

    uncertainty_index = np.argsort(pred_entropies)[::-1]
    uncertainty_index = uncertainty_index[np.in1d(uncertainty_index,unlabeled_indices)][:5]

    delete_indices = np.array([])

    if i %2 ==0 :
        f.text(.05, (1 - (i + 1) * .183),
               "model %d\n\nfit with\n%d labels" %
               ((i + 1), i * 5 + 10), size=10)
    for index,image_index in enumerate(uncertainty_index):
        image = images[image_index]

        if i % 2 ==0 :
            sub = f.add_subplot(5,5,index+1+(5*i))
            sub.imshow(image,cmap=plt.cm.gray,interpolation='None')
            sub.set_title("predict:%i\ntrue:%i"%(model.transduction_[image_index],y[image_index]),size=10)

        #delete the labeled index
        delete_index, = np.where(unlabeled_indices == image_index)
        delete_indices +=len(uncertainty_index)
plt.subplots_adjust(left=.2,bottom=.03,right=.9,top=.9,wspace=.2,hspace=.85)
plt.show()







#
#         # labeling 5 points, remote from labeled set
#         delete_index, = np.where(unlabeled_indices == image_index)
#         delete_indices = np.concatenate((delete_indices, delete_index))
#
#     unlabeled_indices = np.delete(unlabeled_indices, delete_indices)
#     n_labeled_points += len(uncertainty_index)
#
# f.suptitle("Active learning with Label Propagation.\nRows show 5 most "
#            "uncertain labels to learn with the next model.", y=1.15)
# plt.subplots_adjust(left=0.2, bottom=0.03, right=0.9, top=0.9, wspace=0.2,
#                     hspace=0.85)
# plt.show()