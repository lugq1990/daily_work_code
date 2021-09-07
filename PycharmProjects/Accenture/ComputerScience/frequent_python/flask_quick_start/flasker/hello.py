# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""
from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello():
    return "hello world"

