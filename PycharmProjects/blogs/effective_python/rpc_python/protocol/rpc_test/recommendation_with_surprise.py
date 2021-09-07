from surprise import Dataset, evaluate
from surprise import KNNBasic, SVD, Reader
import os
import numpy as np
import pandas as pd


path = r"C:\Users\guangqiiang.lu\Documents\lugq\Kaggle\MovieLens\ml-100k"

file_name = "u.data"


data = Dataset.load_builtin('ml-100k')

reader = Reader(rating_scale=(.5, 5.))

model = SVD()

evaluate(model, data, measures=['RMSE'])


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import log_loss, r2_score

