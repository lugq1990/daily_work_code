# -*- coding:utf-8 -*-
from flask import Flask, request, jsonify
import numpy as np
from sklearn.externals import joblib
import json

# Here load model from disk
model_path = 'C:/Users/guangqiiang.lu/Documents/lugq/workings/201811/flask_demo'
model = joblib.load(model_path+ '/lr.pkl')

# Here make one app
app = Flask(__name__)

# Here make one app request for post response
# Route is api service and methods are post response
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Use flask to get data from requst using json format
    data = request.get_json(force=True)
    # Then extract data from this json format data
    data = np.array([data['d1'], data['d2'], data['d3'], data['d4']])
    # Because of maybe multi-data will be getted in one time, convert them for model accepted format

    # For now, just one sample
    data = data.reshape(1, -1)

    # Use loaded model to do prediction job, in case of int type error, convert it to int64
    pred = [int(model.predict(data))]

    # return the result response using jsonify
    return jsonify(results=json.dumps({'prediction':pred}))


# Start the service
if __name__ == '__main__':
    app.run(port=9000, debug=True)