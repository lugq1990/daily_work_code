"""Test with flink stateful functions"""
import json
from statistics import mean
from pandas import value_counts
from statefun import *
import asyncio

from sympy import re
from aiohttp import web

functions = StatefulFunctions()


greet_request_type = make_json_type(typename="example/GreetRequest")
egress_record_type = make_json_type(typename="io.statefun.playground/EgressRecord")


@functions.bind(typename="example/person",
                specs=[ValueSpec(name='visits', type=IntType)])
async def persion(context:Context, message: Message):
    visits = context.storage.visits or 0
    visits += 1
    context.storage.visits = visits
    
    request = message.as_type(greet_request_type)
    request['visits']= visits
    
    context.send(message_builder(
        target_typename="example/greeter",
        target_id=request['name'],
        value = request,
        value_type=greet_request_type
    ))
    

@functions.bind(typename='example/greeter')
async def greeter(context, message):
    request = message.as_type(greet_request_type)
    
    person_name = request['name']
    visits = request['visits']
    
    greeting = await compute_fancy_greeting(person_name, visits)
    
    egress_record = {
        "topic": "greetings",
        "payload": greeting
    }
    
    context.send_exgress(
        egress_message_builder(target_typename="io.statefun.playground/egress",
                                                value=egress_record,
                                                value_type=egress_record_type))
    
async def compute_fancy_greeting(name: str, seen:int):
    templates = ["", "Welcome %s", "Nice to see you again %s", "Third time is a charm %s"]
    if seen < len(templates):
        greeting = templates[seen] % name
    else:
        greeing = f"Nice to see you at the {seen}-nth time {name}!"
    await asyncio.sleep(1)
    return greeing


handler = RequestReplyHandler(functions)

async def handle(request):
    req = await request.read()
    res = await handler.handle_async(req)
    return web.Response(body=res, context_type="application/octet-stream")


app = web.Application()
app.add_routes([web.post('/statefun', handle)])

if __name__ == "__main__":
    web.run_app(app, port=8000)