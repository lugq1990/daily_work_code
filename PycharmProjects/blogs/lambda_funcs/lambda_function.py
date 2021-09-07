"""Just test with lambda functions, so here if we need to deploy cloud functions,
so need to create a requirements.txt by ourself, also we need to download the packages manually by:
`python -m pip install -r requirements.txt -t .` to download needed packages local, 
then we need to zip them into a zip file and upload it into functions.
Or we could follow steps in official website to update function with dependencies:https://docs.aws.amazon.com/zh_cn/lambda/latest/dg/python-package-update.html
"""
import json
import requests


def lambda_handler(event, context):
    # TODO implement
    d = requests.get("https://www.baidu.com")

    return {"code": d.status_code, 'body': json.dumps('Hello from Lambda!')}

