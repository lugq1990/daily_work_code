from sanic import Sanic, Blueprint
from sanic.response import text, redirect, json

# app = Sanic(__name__)
bp = Blueprint('my')

@bp.route('/bp')
async def bp_root(request):
    return json({"start": "bp"})

app = Sanic(__name__)

app.blueprint(bp)


# @app.route('/')
# async def test(request):
#     return text("hello world")

# @app.route('/hi')
# async def hi(request):
#     return text("hi")


@app.route('/post', methods=['post'])
async def post(request):
    return text("post - {}".format(request.json))


@app.route('/get', methods=['get'])
async def get(request):
    return text("get {}".format(request.args))

@app.route('/')
async def index(request):
    url = app.url_for("post_handler", post_id=0)
    return redirect(url)

@app.route('/posts/<post_id>')
async def post_handler(request, post_id):
    return text("post -> {}".format(post_id))

app.run(host='0.0.0.0', port=8000, debug=True)