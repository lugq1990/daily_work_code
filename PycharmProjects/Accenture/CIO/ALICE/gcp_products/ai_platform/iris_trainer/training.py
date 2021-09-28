import datetime
import os
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from joblib import dump, load

x, y = load_iris(return_X_y=True)
lr = LogisticRegression()

lr.fit(x, y)

# dump model into disk
dump(lr, 'model.joblib')
