# -*- coding:utf-8 -*-
"""This class is used to validate for parameters choosen by setting param_name and params,
    also can be used to get different thresholds for model prediction validation box plot curve.
    This also support to use user defined metric function to evaluate CV scoring.
 """
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style

class plotDiffParamBox(object):
    def __init__(self):
        pass

    def plotDiffBox(self, estimator, x, y, param_name=None, params=None, cv=10, cut_off=True, cut_range=None, metric=None,
                    figsize=(8, 6), title=None):
        style.use('ggplot')

        scores = list()
        inx = list()
        x, y, estimator = x, y, estimator

        if params is None and not cut_off:
            raise AttributeError('Must choosen for using params choosen or cutoff for probalility!')

        # this is used for probability different threshold prediction plot.
        if cut_off and params is None:
            if len(np.unique(y)) != 2:
                raise ValueError('For using cut_off, label y must be Binary!')

            if not hasattr(estimator, 'predict_proba'):
                raise ValueError('Estimator must have predict_proba funcion!')

            # this is for different threshold for predicting function
            def _cut_off(estimator, x, thre):
                return (estimator.predict_proba(x)[:, 1] > thre).astype(int)

            # this is for computing score for estimator
            def _score_f(thre):
                def _sc(estimator, x, y):
                    # if metric is not given, just use 'accuracy' for scoring for estimator
                    if metric is None:
                        return metrics.accuracy_score(y, _cut_off(estimator, x, thre=thre))
                    else:
                        # use user given metric function.
                        return metric(y, _cut_off(estimator, x, thre=thre))
                return _sc

            # if user does not give the cut_range parameter, then just use 10 folds for 0-1
            if cut_range is None:
                cut_range = np.arange(0, 1., .1)

            # loop for cut_range, here use 'cv' folds cross-validation evaluation.
            for thre in cut_range:
                scores.append(cross_val_score(estimator, x, y, cv=cv, scoring=_score_f(thre)))

            # loop for index for bellowing dataframe construction, because for 'cv' cross-validation index,
            # here I just use not smart function to construct the index(for loop for constructing the index)
            for i in range(len(cut_range)):
                # t = np.ones(cv)*round(cut_range[i], 2)  ## get round 2 point, not using for loop to construct index
                inx.append([round(cut_range[i], 2)] * cv)
        else:
            if params is None or param_name is None:
                raise ValueError('For params choosen, params and param_name must be given')

            # this is for sklearn estimator set estimator parameters
            for p in params:
                estimator.set_params(**{str(param_name): p})
                if metric is not None:
                    # for sklearn metric object, if want to use it as cross-validation scoring,
                    # must use make scorer to convert the metric object to scoring object
                    from sklearn.metrics import make_scorer
                    converted_metric = make_scorer(metric)
                    scores.append(cross_val_score(estimator, x, y, cv=cv, scoring=converted_metric))
                else: scores.append(cross_val_score(estimator, x, y, cv=cv))

            # loop for params, make the index list
            for i in range(len(params)):
                # t = np.ones(len(cv)) * params[i]
                # in case choosen params is string type, here construct index using for loop(Not Doing this)
                # there is a more efficient way to do this
                inx.extend([params[i]] * cv)

        merge_res = np.concatenate((np.array(inx).reshape(-1, 1), np.array(scores).reshape(-1, 1)), axis=1)
        merge_res = pd.DataFrame(merge_res)
        # this DataFrame is 2 columns, first is index cols as same key for each parameter or each cut_range,
        # indexed value is cross-validation scores.
        merge_res.columns = ['inx', 'score']

        # Plot the box plot curve
        sns.boxplot(x=merge_res.inx, y=merge_res.score)

        if title is not None:
            plt.title(title)
        else:
            if cut_off and params is None:
                if metric is not None:
                    plt.title("Different Cut_off range For '%s' Box Curve" % (metric.__name__))
                else: plt.title("Different Cut_off range For '%s' Box Curve" % ('Accuracy'))
                plt.xlabel("Cut_off Range")
                plt.ylabel("Scoring")
                plt.xticks(np.arange(len(cut_range)), np.unique(merge_res.inx))
            else:
                if metric is not None:
                    plt.title("Different Params '%s' For '%s' Box Curve" % (str(param_name), metric.__name__))
                else: plt.title("Different params '%s' For '%s' Box Curve" %(str(param_name), 'Accuracy'))
                plt.xlabel("Params '%s' "%str(param_name))
                plt.ylabel("Scoring")
                plt.xticks(np.arange(len(params)), params)

        plt.legend()
        plt.show()


if __name__ == '__main__':
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.datasets import load_iris
    from sklearn import metrics
    iris = load_iris()
    # x, y = iris.data[:100, :], iris.target[:100]
    # x, y = iris.data, iris.target
    x = np.random.random((1000, 5))
    y = np.random.randint(2, size=(1000,))

    lr = LogisticRegression()
    clf = SVC(probability=True)

    plotDiffParamBox().plotDiffBox(clf, x, y, param_name='C', params=[10, 100], cut_off=False)

