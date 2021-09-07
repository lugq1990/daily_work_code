# -*- coding:utf-8 -*-
"""
This is based on the dash library to plot some interative plot

@author: Guangqiang.lu
"""
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime
import random
from collections import deque
import plotly
import plotly.graph_objects as go


# iris = load_iris()
# x, y = iris.data, iris.target
# feature_names = list('abcd')
# feature_names.extend(['label', 'label_name'])
# target_name = iris.target_names
# target_names = []
# for i, n in enumerate(target_name):
#     target_names.extend([n] * 50)
#
# data = np.concatenate([x, y[:, np.newaxis], np.array(target_names)[:, np.newaxis]], axis=1)
# df = pd.DataFrame(data, columns=feature_names)
#
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#
# markdown_text = '''
# ### Dash and Markdown
#
# Dash apps can be written in Markdown.
# Dash uses the [CommonMark](http://commonmark.org/)
# specification of Markdown.
# Check out their [60 Second Markdown Tutorial](http://commonmark.org/help/)
# if this is your first introduction to Markdown!
# '''
#
# app.layout = html.Div([
#     dcc.Markdown(children=markdown_text),
#     dcc.Graph(id='iris',
#               figure={
#                   'data':[
#                       dict(x=df[df['label']==i]['a'], y=df[df['label']==i]['b'],
#                            text=df[df['label']==i]['label_name'],
#                            mode='markers', opacity=.7, marker={
#                               'size': 15,
#                               'line': {'width': .5, 'color': 'white'},
#
#                           },
#                            name=i) for i in df.label.unique()
#                   ],
#                   'layout': dict(xaxis={'type': 'log', 'title': 'IRIS'},
#                              yaxis={'title': 'length'},
#                              margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
#                              legend={'x': 0, 'y':  1},
#                              hoveromde='closest')
#               })
# ])


# This is interative with user
# app = dash.Dash()
#
# app.layout = html.Div(children=[
#     dcc.Input(id='input', value='put some data', type='text'),
#     html.Div(id='output')
# ])
#
#
# @app.callback(
#     Output(component_id='output', component_property='children'),
#     [Input(component_id='input', component_property='value')]
# )
# def update_value(input_value):
#     try:
#         return str(float(input_value) ** 2)
#     except:
#         return "Error faced."


# This could be used to get the user input and dynamic draw the graph
# app = dash.Dash()
#
# app.layout = html.Div(children=[
#     html.Div(children="Diff company stock price"),
#     dcc.Input(id='input', value='', type='text'),
#     html.Div(id='output')
# ])
#
# @app.callback(
#     Output(component_id='output', component_property='children'),
#     [Input(component_id='input', component_property='value')]
# )
# def get_graph(input_value):
#     start_time = datetime.datetime(2015, 1, 1)
#     end_time = datetime.datetime.now()
#
#     df = web.DataReader(input_value, 'yahoo', start_time, end_time)
#
#     return dcc.Graph(
#         id='stock_exa',
#         figure={
#             'data':[
#                 {'x': df.index, 'y': df.Close, 'type': 'line', 'name': input_value},
#             ],
#             'layout':{
#                 'title': input_value
#             }
#         }
#     )


# Here is to test with different core components
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div(children=[
    # dcc.Dropdown(
    #     options=[
    #         {'label': 'lu', 'value': 'lugq'},
    #         {'label': 'liu', 'value': 'luiyu'}
    #     ],
    #     multi=True,
    #     value='lugq'
    # )

    # slider
    # dcc.Slider(min=4, max=10, step=.5, value=5,
    #            marks={i: "label {}".format(i) for i in range(4, 10)})

    # range slider
    dcc.RangeSlider(min=-5, max=5, step=1, value=[-3, 0],
                    marks={i: "label {}".format(i) for i in range(-5, 5)}),

    dcc.Input(placeholder='input text', type='text', value=''),

    dcc.Textarea(placeholder='some text', value='default', style={"width":'100%'}),

    dcc.Checklist(options=[
                {'label': 'lu', 'value': 'lugq'},
                {'label': 'liu', 'value': 'luiyu'},
                {'label': 'mei', 'value': 'meiyangyang'},
    ],
    value=['lu', 'liu'], labelStyle={'display': 'inline-block'}),

    dcc.RadioItems(options=[
                {'label': 'lu', 'value': 'lugq'},
                {'label': 'liu', 'value': 'luiyu'},
                {'label': 'mei', 'value': 'meiyangyang'},
    ],
    value='lugq', labelStyle={'display': 'inline-block'}),

    dcc.DatePickerSingle(id='date', date=datetime.datetime(2000, 1, 1)),

    dcc.DatePickerRange(id='select_date', start_date=datetime.datetime(2000, 1, 1), end_date_placeholder_text='enter here'),

    dcc.Markdown('''
    #### Dash and Markdown
    
    Dash supports [Markdown](http://commonmark.org/help).
    
    Markdown is a simple way to write and format text.
    It includes a syntax for things like **bold text** and *italics*,
    [links](http://commonmark.org/help), inline `code` snippets, lists,
    quotes, and more.
    '''),

    dcc.ConfirmDialogProvider(children=html.Button('click me'), id='confirm', message='are you sure?')
])

# app.layout = html.Div(children=[
#     html.Div(dcc.Input(id='input-text', type='text')),
#     html.Button('Submit', id='button'),
#     html.Div(id='button-output', children='enter a value')
# ])
#
#
# @app.callback(Output('button-output', 'children'),
#               [Input('button', 'n_clicks')],
#               [State('input-text', 'value')])
# def update_output(n_clicks, value):
#     return "Input {} get {} clicks".format(value, n_clicks)


if __name__ == '__main__':
    app.run_server(debug=True)
