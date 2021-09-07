# -*- coding:utf-8 -*-
from flask import Flask

app = Flask("hello_world")

# @app.route('/hello')
# def hello_world():
#     return 'This is first hello world using flask'
#
# @app.route('/index')
# def index():
#     return 'This is to index'

# @app.route('/user/<username>')
# def show_user_profile(username):
#     return 'This is {}'.format(username)
#
# @app.route('/post/<int:post_id>')
# def show_post(post_id):
#     return 'Post {}'.format(post_id)
#
# @app.route('/path/<path:subpath>')
# def show_subpath(subpath):
#     return 'Get subpath {}'.format(subpath)

# from flask import url_for
#
# @app.route('/')
# def index():
#     return 'This is index'
#
# @app.route('/login')
# def login():
#     return 'this is to login'
#
# @app.route('/user/<username>')
# def profile(username):
#     return 'hello {}'.format(username)
#
# with app.test_request_context():
#     print(url_for('index'))
#     print(url_for('login'))
#     print(url_for('login', next='/'))
#     print(url_for('profile', username='lu guangqiang'))

# from flask import abort, url_for, redirect, request
# #
# # @app.route('/')
# # def index():
# #     return redirect(url_for('login', username='guanqqiang'))
# #
# # ## for the request object is used for post method
# # @app.route('/login/<username>')
# # def login(username):
# #     return 'how are you? {}'.format(username)


### this is to get the value from Session object, for username with store the value in the session object.
# here also with redirect to another page
from flask import url_for, session, escape, request, redirect
import os
sec = os.urandom(16)

# set the secret key
app.secret_key = sec

@app.route('/')
def index():
    if 'username' in session:
        app.logger.debug('add username to the session')
        return 'Logged as {}'.format(escape(session['username']))
    return 'You are not login'

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['username'] = request.form['username']
        return redirect(url_for('index'))
    return '''
        <form method="post">
            <p><input type=text name=username>
            <p><input type=submit value=Login>
        </form>
    '''

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(port=5000, debug=True)