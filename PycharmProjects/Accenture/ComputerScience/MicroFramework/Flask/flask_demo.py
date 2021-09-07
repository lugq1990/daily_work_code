"""This is used to show some cases that could be used
for web applications, as flask is a lite module for web,
so we could really get start with easy step.
First you should install with pip:
pip install flask
Then most of this modules are based on the official website tutorials:
https://flask.palletsprojects.com/en/1.1.x/quickstart/
Here I will show you.
"""

from flask import Flask, escape, url_for, request, redirect, \
    jsonify, render_template, make_response, session
import time
import datetime

app = Flask(__name__)

# 1. most basic example

# @app.route('/')
# def hello_world():
#     return "Hello world!"

# -----------
# 2. Route
# -----------
# @app.route('/')
# def index():
#     return "This is index page"
#
#
# @app.route('/hello')
# def hello():
#     return "Hello worlds, just test!"

# ---------
# 3. variables: in fact we could add some variables to the URL for querying something
# ---------
# @app.route('/user/<username>')
# def show_user_frofile(username):
#     return "user %s" % username
#
#
# @app.route('/post/<int:post_id>')
# def show_post(post_id):
#     return "Get Post %d" % post_id
#
#
# @app.route('/path/<path:subpath>')
# def show_subpath(subpath):
#     return "Get subpath: %s" % subpath


# ----------
# 4. unique URL: we could add / or not to separate with each resource
# ----------
# @app.route('/projects/')
# def projects():
#     # this is just the path of a file
#     return "This is projects page"
#
#
# @app.route('/about')
# def about():
#     # this is like the path of a file, so if I get the function with /about/ will raise not found error
#     # but we could still get the page with /about URL
#     return "This is about page"

# ----------
# 5. build our URL: There are few benefits that we should use url_for() function to build URL:(1). it's more descriptive;
# (2). We don't need to use hard-code urls; (3) we could handle with /about path like before
# ----------
# @app.route('/')
# def index():
#     return "index page"
#
#
# @app.route('/login')
# def login():
#     return "Login page"
#
#
# @app.route('/user/<username>')
# def profile(username):
#     return "{}\'s profile".format(escape(username))
#
#
# # with test to make the url
# with app.test_request_context():
#     print(url_for('index'))
#     print(url_for('login'))
#     print(url_for('login', next='/'))
#     print(url_for('profile', username='lu'))


# --------
# 6. HTTP method: there are two methods are used most in HTTPS(just safer than HTTP): GET and POST,
# the difference between GET and POST is that POST is more safer than GET, and GET has to pass whole
# parameters in URL: like /username/lugq... but with POST we could just like this: /username/change...
# this link for difference is great: https://stackoverflow.com/questions/3477333/what-is-the-difference-between-post-and-get/3477374#3477374
# without any sensitive data, right? so POST method is used most!
# for HTTP also has 'DELETE'...
# --------
# username_globel = None
#
# @app.route('/login/<username>', methods=['GET', 'POST'])
# def login(username):
#     if request.method == 'GET':
#         return get_name(username)
#     else:
#         # use request to get name
#         # but how could we do post action by our hands for example?
#         # requests.post("http://127.0.0.1:5000/login/username", json={'username':'lugq'}).content
#         # you have to remember that for post method is used in change data with backend...
#         # when we get one page, what we use is just GET method!
#         username_globel = request.json['username']
#         return do_login(username=username_globel)
#
#
# def get_name(username):
#     return "Get name: {}".format(username)
#
#
# def do_login(username):
#     return jsonify({"Get_name": username})

# ----------
# 7. Global variables: as sometimes we do need to react to user's request, so sometimes we have to
# store data with variables, so here we should use request object to store the data.
# But how could we do post? use this:
# requests.post("http://127.0.0.1:5000/login", data={'username':'lu', 'password':'12'}).content
# ----------
# @app.route('/hello', methods=['GET', 'POST'])
# def hello():
#     return "Hello world"
#
#
# # here we could just do a unit test like test for request
# # after add this, if we use post method, we could find with error
# with app.test_request_context('/hello', method='POST'):
#     assert request.path == '/hello'
#     assert request.method == 'POST'
#
#
# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     error = None
#     if request.method == 'POST':
#         if validate(request.form['username'], request.form['password']):
#             return success(request.form['username'])
#         else:
#             error = 'invalidate username or password'
#             return error
#     else:
#         return "Just a get method"
#
#
# def validate(username, password):
#     if username is None or password is None:
#         return False
#     return True
#
#
# def success(username):
#     return "You are allowed: {}".format(username)

# ----------
# 8. upload file with flask: sometimes we want to use web to upload file to server or remote
# cloud server, so here is just to show you how to upload file
# ----------

# from flask import Flask, abort, render_template, request, redirect, url_for
# from werkzeug.utils import secure_filename    # we should use this to ensure we get the right file name
# import os
#
# upload_path = "C:/Users/guangqiiang.lu/Documents/lugq/PycharmProjects/Accenture/ComputerScience/MicroFramework/Flask/upload_folder"
# app.config['upload_path'] = upload_path
#
# @app.route('/')
# def index():
#     return redirect(url_for('hello'))
#
# @app.route('/hello')
# @app.route('/hello/<name>')
# def hello(name=None):
#     return render_template('hello.html', name=name)
#
#
# @app.route('/upload/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         file = request.files['file[]']
#         if file:
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['upload_path'], filename))
#             # here could return with hello world page!
#             return hello()
#     return render_template('file_upload.html')

# ------------
# 9. redirection and errors: sometimes we want user click one button, then
# we want users to see another page, so we should use redirection function,
# and we could also make the user stop early with error.
# -------------

# @app.route('/')
# def index():
#     return redirect(url_for('login'))
#
#
# @app.route('/login/<username>')
# def login(username):
#     if username is not None:
#         curr_date = datetime.datetime.now()
#         res = {"username": username, "login_time": curr_date}
#         # here if we want to return the dictionary result, we have to use jsonify.
#         return jsonify(res)
#     else:
#         return page_not_found()
#
#
# @app.errorhandler(404)
# def page_not_found(error):
#     # you have to give the function with error key words.
#     # for error 404 means couldn't find that page,
#     # if success, the return code is 200
#     # we could also make a response with error
#     resp = make_response(render_template('page_not_found.html'), 404)
#     resp.headers['Add'] = 'test'
#     return resp

# ---------
# 10. Session object: sometimes we not only want to use request to get the object
# that we get, we also want to pass the whole info of user login information for later
# use case, so here just with one example
# ---------

# one thing should be first noticed is we should set a secret key for each program!
app.secret_key = b'dfhsldkfowe'


@app.route('/')
def index():
    if 'username' in session:
        return "Login as {}".format(session['username'])
    else:
        return "You are not login!"


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


@app.route("/logout")
def logout():
    # we could also add logger with flask, in fact flask will just use standard logging module
    app.logger.info("User logout")
    # remove the username from the session object
    session.pop('username', None)
    return redirect(url_for('index'))


if __name__ == '__main__':
    # with debug to True, I don't need to stop and start again and again.
    app.run(debug=True)