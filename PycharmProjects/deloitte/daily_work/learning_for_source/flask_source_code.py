from flask import Flask, json

app = Flask(__name__)

@app.route('/hello')
def hello():
    return json.dumps({"greeting": 'hello flask'})


if __name__ == '__main__':
    app.run(debug=True)