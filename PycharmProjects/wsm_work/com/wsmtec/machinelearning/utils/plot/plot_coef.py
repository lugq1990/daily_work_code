# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt

def plot_coef(estimator):
    coef = estimator.coef_.reshape(-1,1)
    plt.figure(1,figsize=(6,4))
    plt.clf()
    plt.plot(coef,linewidth=2)
    plt.axis('tight')
    plt.title('Logistic Regression Coef')
    plt.xlabel('Variables')
    plt.ylabel('Coef')
    plt.show()