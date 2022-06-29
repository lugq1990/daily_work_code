import asyncio
from sanic import Sanic
from sanic.response import text, json, redirect
from ujson import dumps
from sanic.views import HTTPMethodView


app = Sanic("hello_sanic")

app.config.DB_NAME = "mysql"
app.config["DB_USER"] = 'root'

db_settings = {
    "db_host": '127.0.0.1',
    "db_name": "mysql",
    "db_user":"root",
    "db_password":1234
}

app.config.update(db_settings)

@app.reload_process_start
async def reload_start(*_):
    print("<<<< start >>>>")

@app.main_process_start
async def main_start(*_):
    print(">>>> start server <<<<")

@app.main_process_stop
async def main_stop(*_):
    print(">>>> stop <<<<")
# @app.get("/")
# async def hello_world(request):
#     return text("hello, this is sanic!")

@app.get("/foo")
async def foo_handler(request):
    return text("this is foo")

@app.on_request
async def increment_foo(request):
    if not hasattr(request.conn_info.ctx, "foo"):
        request.conn_info.ctx.foo = 0
    request.conn_info.ctx.foo += 1
    
    
@app.get("/")
async def count_foo(request):
    # return text(f"foo={request.conn_info.ctx.foo}")
    return json({"foo": "bar"}, dumps=dumps)

async def handler(request):
    return text("ok")

app.add_route(handler,'/test', methods=['GET', "POST"])


@app.get("/tag/<tag>")
async def tag(request, tag):
    return text("tag -{}".format(tag))

@app.route("/posts/<post_id>", name='get_handler')
async def get(request, post_id):
    return text("ok, get {}".format(post_id))
    

@app.get("/url")
async def url(request):
    url = app.url_for('get_handler', post_id=10)
    return redirect(url)

@app.websocket("/web")
async def handler(request, ws):
    msg = "start"
    while True:
        await ws.send(ws)
        ers = ws.recv()
        print(ers)
        
        
async def notity_server_start():
    await asyncio.sleep(2)
    print("server started!!!!")
    
async def auto_inject(app):
    await asyncio.sleep(5)
    print(app.name)

app.add_task(auto_inject)