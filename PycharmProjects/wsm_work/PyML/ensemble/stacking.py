# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 09:26:54 2017

@author: Administrator
"""
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, classifiers, vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights
        
    def fit(self, X, y):
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, y)
            self.classifiers_.append(fitted_clf)
        return self
        
    def predict(self, X):
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.weights)), axis=1, arr=predictions)
        return maj_vote
        
    def predict_proba(self, X):
        probas  = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba
        
    def get_params(self, deep=True):
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' %(name, key)] = value
        return out
        

from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score, accuracy_score, roc_auc_score, f1_score


class MajorityVoteLogistic(BaseEstimator, ClassifierMixin):
    
    def __init__(self, classifier, n_estimators=10, split_size=.2, boostrap=False, 
                 under_sampling=False, vote='classlabel', seed=0):
        self.classifier = classifier
        self.seed = seed
        self.n_estimators = n_estimators
        self.split_size = split_size
        self.boostrap = boostrap
        self.vote = vote
        self.under_sampling = under_sampling
        
    def fit(self, X, y):
        accu = 0
        recall = 0
        roc = 0
        f1 = 0
        self.classifiers_ = []
        for i in range(self.n_estimators):
            X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=self.split_size)
            
            if not self.under_sampling:
                if self.boostrap:
                    #boostrap sampling
                    random_state = self.seed + i
                    random_state = np.random.RandomState(random_state)
                    indices_train = random_state.randint(0, X_train.shape[0], X_train.shape[0])
                    indices_test = random_state.randint(0, X_test.shape[0], X_test.shape[0])
                    X_train = X_train[indices_train, :]
                    X_test = X_test[indices_test, :]
                    y_train = np.asarray(y_train)[[indices_train]]
                    y_test = np.asarray(y_test)[[indices_test]]
            else:
                #under sampling
                random_state = self.seed + i
                random_state = np.random.RandomState(random_state)
                indices_y1 = np.where(y_train==1)[0]
                indices_y0 = np.where(y_train==0)[0]
                indices_y0 = random_state.choice(indices_y0, len(indices_y1), replace=False)
                X_train0 = np.array(X_train)[indices_y0, :]
                X_train1 = np.array(X_train)[indices_y1, :]
                X_train = np.concatenate([X_train0, X_train1], axis=0)
                y_train0 = np.array(y_train)[indices_y0]
                y_train1 = np.array(y_train)[indices_y1]
                y_train = np.concatenate([y_train0, y_train1], axis=0)
            
            fitted_clf = clone(self.classifier).fit(X_train, y_train)
            self.classifiers_.append(fitted_clf)
            y_pred = fitted_clf.predict(X_test)
            y_predprob = fitted_clf.predict_proba(X_test)[:,1]
            accu += accuracy_score(y_test, y_pred)
            recall += recall_score(y_test, y_pred)
            roc += roc_auc_score(y_test, y_predprob)
            f1 += f1_score(y_test, y_pred)
            #print np.sum(y_test), np.sum(y_pred)
        
        print 
        print "==============================="
        print "Cross_validation Result"
        print "   avg_accur: %f" %(accu/self.n_estimators)
        print "  avg_recall: %f" %(recall/self.n_estimators)
        print "     avg_roc: %f" %(roc/self.n_estimators)
        print "      avg_F1: %f" %(f1/self.n_estimators)
        print "==============================="
        print
        return self
        
    def predict(self, X):
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=predictions)
        return maj_vote
        
    def predict_proba(self, X):
        probas  = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0)
        return avg_proba
        