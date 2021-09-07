# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""
import os
from google.cloud import translate
from flask import Flask
from flask import render_template, make_response, request


app = Flask(__name__)

credential_path = 'C:/Users/guangqiiang.lu/Downloads/google_cloud'
credential_name = 'CloudTutorial-0364634f2133.json'
project_id = 'cloudtutorial-278306'

# set env
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(credential_path, credential_name)

client = translate.TranslationServiceClient()
parent = client.location_path(project_id, 'global')


def translate(text):
    res = client.translate_text(parent=parent,
                                contents=[text],
                                mime_type='text/plain',
                                source_language_code='en-US',
                                target_language_code='zh-CN')

    for t in res.translations:
        yield t.translated_text


@app.route('/home')
def home():
    return render_template('hello.html'), 200


if __name__ == '__main__':
    app.run(debug=True)

