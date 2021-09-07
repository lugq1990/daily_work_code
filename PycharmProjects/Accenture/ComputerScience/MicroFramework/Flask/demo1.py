# -*- coding:utf-8 -*-
from flask import Flask
from flask import jsonify
from flask import request

app = Flask(__name__)

@app.route('/')
def index():
    return 'index page'

@app.route('/hello')
def hello():
    return 'hello world'

@app.route('/about')
def about():
    return 'This is about function'


## This is variable for function
@app.route('/user/<username>')
def show_user(username):
    return 'User %s'%username

@app.route('/post/<int:post_id>')
def show_post(post_id):
    return 'Get post_id is %d'%post_id

@app.route('/path/<path:subpath>')
def show_subpath(subpath):
    return 'Subpath %s'%subpath


### This is test for triggering model training and model testing
from flask import redirect, url_for
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

x, y = load_iris(return_X_y=True)
lr = LogisticRegression()
@app.route('/model/train/<float:C>', methods=['GET', 'POST'])
def model_train(C):
    if request.method == 'GET':
        # param_c = request.args.get('C')
        lr.C = C
        lr.fit(x, y)
        return jsonify({'status': 'training finished!'})
        # This is used for redirect the url for test
        # return redirect(url_for('model_test'))
    if request.method == 'POST':
        # data =request.get_json(force=True)
        # param_c = request.args.get('C')
        param_c = request.form.get('C')
        lr.C = param_c
        lr.fit(x, y)
        score = lr.score(x, y)
        r = request.post('127.0.0.1/5000/re', json={'training score': score})
        return jsonify({'status': 'Finished model training', 'training score': score}), 202


@app.route('/model/test')
def model_test():
    acc = lr.score(x, y)
    acc_dic = {'accuracy': acc}
    return jsonify(acc_dic)
    # return 'model test accuracy: %s'%(str(acc))



# TEST for HTTP get and response

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         return 'Please login'
#     else:
#         return 'Show some infomation'
#
# # Redirect and errors
# from flask import redirect, abort, url_for
# @app.route('/index_test')
# def index_test():
#     return redirect(url_for('login_test'))
# @app.route('/login_test')
# def login_test():
#     abort(401)
#     return 'This is login test funtion'


# Flask session object
# from flask import Flask, session, redirect, url_for, escape, request
# import os
# app = Flask(__name__)
#
# app.secret_key = os.urandom(16)
#
# @app.route('/index_test')
# def index_test():
#     if 'username' in session:
#         return 'Login as % s'% escape(session['username'])
#     return 'you are Not login!'
# @app.route('/login_test', methods=['GET', 'POST'])
# def login_test():
#     if request.method == 'POST':
#         session['username'] = request.form['username']
#         return redirect(url_for('index_test'))
#     return '''
#         <form method="post">
#             <p><input type=text name=username>
#             <p><input type=submit value=Login>
#         </form>
#     '''
# @app.route('/logout')
# def logout():
#     session.pop('username', None)
#     print('You have log not!')
#     return redirect(url_for('index_test'))
#


if __name__ == '__main__':
    app.run()