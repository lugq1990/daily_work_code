import warnings

warnings.simplefilter('ignore')

import time
from sklearn.datasets import load_iris, load_digits
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression,  LinearRegression
from sklearn.ensemble import RandomForestClassifier



lr = LogisticRegression()
li = LinearRegression()
rfc = RandomForestClassifier()

model_list = [lr, li, rfc]

x, y = load_digits(return_X_y=True)


def fit(model):
    try:
        model.fit(x, y)
        print("Done")
        print("Model score:", model.score(x,  y))
    except Exception as e:
        print("get error: ", e)


def p():
    for model in model_list:
        model.fit(x, y)
        print("Model score:", model.score(x,  y))


if __name__ == "__main__":
    start_time = time.time()
    res = Parallel(n_jobs=3)(delayed(fit)(model) for model in model_list)
    print("Parallel time: ", time.time() - start_time)

    start_time = time.time()
    p()
    print("Not Parallel time: ", time.time() - start_time)
