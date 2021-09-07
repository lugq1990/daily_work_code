# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""
from flask import Flask, url_for, request
from markupsafe import escape

app = Flask(__name__)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return do_login()
    else:
        return show()

def do_login():
    return 'We have logined'

def show():
    return 'now to show'


@app.route('/save', methods=['GET', 'POST'])
def save_file():
    if request.method == 'POST':
        f = request.files['the_file']
        import os
        abs_path = os.path.abspath(os.curdir)
        print("Cur dir:" + abs_path)
        f.save(os.path.join(abs_path, 'new_file.txt'))
        return "good news."


if __name__ == '__main__':
    app.run(debug=True)