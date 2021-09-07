# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""
def hello_world(request):
    json_request = request.get_json()

    if "message" in request.args:
        return request.args.get("message") + " , so good!"
    elif json_request and "message" in json_request:
        return json_request['message']
    else:
        return f"not so good..."