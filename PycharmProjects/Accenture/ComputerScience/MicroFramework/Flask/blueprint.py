# -*- coding:utf-8 -*-
from flask import Flask, Blueprint

app = Flask(__name__)

# Start to build the blueprint template
# This is based on a tree mode
tree_mode = Blueprint('mold', __name__)

@tree_mode.route('/leaves')
def leaves():
    return 'This tree has leaves'

@tree_mode.route('/root')
def root():
    return 'This tree has root'


@tree_mode.route('/rings')
@tree_mode.route('/rings/<int:year>')
def rings(year=None):
    return 'This tree has rings:{year}'.format(year=year)


# Because if we not register the function, then we can't use it
app.register_blueprint(tree_mode, url_prefix='/oak')
app.register_blueprint(tree_mode, url_prefix='/fir')
app.register_blueprint(tree_mode, url_prefix='/ash')


if __name__ == '__main__':
    app.run()



