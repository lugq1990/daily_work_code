# -*- coding:utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import json
import requests

model_path = 'C:/Users/guangqiiang.lu/Documents/lugq/workings/201811/flask_demo'

def model_training():
    has_model = joblib.load(model_path+'/lr.plk')
    if not has_model:
        iris = load_iris()
        x, y = iris.data, iris.target
        lr = LogisticRegression()
        lr.fit(x, y)

        # Dump the model to disk
        joblib.dump(lr, model_path+'/lr.plk')
        print('All finished!')
    else:
        print('Model already trained')

def json_data(data, name_list):
    if not isinstance(data, list):
        try:
            data = list(data)
        except:
            print('Not supported data format!')
    re = dict()
    if len(data) != len(name_list):
        raise AttributeError('Data and name must be same length!')

    for i in range(len(data)):
        re[name_list[i]] = data[i]

    return json.dumps(re)

# This is the basic request function
def post(url, data_dic):
    return requests.post(url, data_dic)

if __name__ == "__main__":
    # Where or to train model?
    model_training()
    data_list = [10, .1, 30, 100]
    name_list = ['d1', 'd2', 'd3', 'd4']
    data = json_data(data_list, name_list)

    # This is service url
    url = "http://127.0.0.1:9000/api"
    re = post(url, data)
    print('Using post gets result:', re.json())

