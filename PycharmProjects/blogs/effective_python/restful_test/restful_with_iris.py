import os
import numpy as np
from sklearn.datasets import load_iris
from joblib import dump, load
from flask import Flask, request, jsonify, render_template
from flask_restful import Resource, Api

from sklearn.linear_model import LogisticRegression

x, y = load_iris(return_X_y=True)

model_path = r"C:\Users\guangqiiang.lu\Documents\lugq\code_for_future\blogs\effective_python\restful_test\static"

if not os.path.exists(model_path):
    raise FileNotFoundError("folder: {} doesn't exist!".format(model_path))


app = Flask(__name__, template_folder='templates')
api = Api(app)



def train_model():
    model_name = os.path.join(model_path, 'lr.pkl')

    if not os.path.exists(model_name):
        lr = LogisticRegression()
        lr.fit(x, y)
        dump(lr, model_name)
    else:
        print("Model has been trained")
        lr = load(model_name)
    
    return lr



@app.before_first_request
def load_model():
    app.predictor = train_model()


@app.route('/')
def index():
    return render_template('index.html', pred=0)


@app.route('/predict', methods=['post'])
def predict():
    data = [request.form['a'], request.form['b'], request.form['c'], request.form['d']]

    data = np.array(data, dtype=float)

    data = data.reshape(1, -1)
    print("get data:",data)

    pred = app.predictor.predict(data)[0]

    print("Get model prediction:", pred)
    
    return render_template("index.html", pred=pred)


# api.add_resource(Predict, '/predict')


if __name__ == '__main__':
    app.run(port=5000, debug=True)
