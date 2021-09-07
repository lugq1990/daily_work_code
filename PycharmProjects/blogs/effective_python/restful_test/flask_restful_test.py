
from flask import Flask
from flask.globals import request
from flask_restful import Resource, Api, reqparse, fields, marshal_with, abort

app = Flask(__name__)
api =  Api(app)

resouce_field = {
    'task': fields.String,
    'uri': fields.Url("todo_ep")
}


class TodoDao:
    def __init__(self, todo_id, task):
        self.todo_id = todo_id
        self.task = task

        self.status = "active"


class Todo(Resource):
    @marshal_with(resouce_field)
    def get(self, **kwargs):
        return Todo(todo_id='my_todo', task="please rememer")



class Hello(Resource):
    def get(self):

        return {"data":"hi restful world!"}

todos = {}


class TodoSimple(Resource):
    def get(self, todo_id):
        return {todo_id: todos[todo_id]}

    def post(self, todo_id):

        todos[todo_id] = request.form['data']
        return {todo_id: todos[todo_id]}

    
class Todo1(Resource):
    def get(self):
        return {"Task": "hi"} 


class Todo2(Resource):
    def get(self):
        return {"Task": "hi"}, 201

class Todo3(Resource):
    def get(self):
        return {"Task": "hi"}, 201, {"Etag": "good news"}


TODOS = {
    'todo1': {'task': 'build an API'},
    'todo2': {'task': '?????'},
    'todo3': {'task': 'profit!'},
}

def abort_if_not_exist(todo_id):
    if not todo_id.startswith('todo'):
        todo_id = 'todo{}'.format(todo_id)
    if todo_id not in TODOS:
        abort(404, message="Todo {} doesn't exist".format(todo_id))


parser = reqparse.RequestParser()
parser.add_argument("task")


class TODO(Resource):
    def get(self, todo_id):
        abort_if_not_exist(todo_id)
        return TODOS[todo_id]
    
    def delete(self, todo_id):
        abort_if_not_exist(todo_id)
        del TODOS[todo_id]
        return 'delet finish', 204
    
    def put(self, todo_id):
        args = parser.parse_args()
        task = {"task": args['task']}
        TODOS[todo_id] = task
        return task, 201


class TODOLIST(Resource):
    def get(self):
        return TODOS

    def post(self):
        args = parser.parse_args()
        todo_id = int(max(TODOS.keys()).lstrip('todo')) + 1
        todo_id = 'todo{}'.format(todo_id)
        TODOS[todo_id] = {'task': args['task']}

        return TODOS[todo_id], 201


import json

class GetData(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        data = json.loads(json_data)['data']

        print("Get data:", data)

        return "Get data!"
# api.add_resource(Hello, "/", '/hello')
# api.add_resource(TodoSimple, '/<string:todo_id>')
# api.add_resource(Todo1, '/todo1')
# api.add_resource(Todo2, '/todo2')
# api.add_resource(Todo3, '/todo3')
# api.add_resource(Todo, '/todo_test')
api.add_resource(TODOLIST, '/todos')
api.add_resource(TODO, '/todos/<todo_id>')

api.add_resource(GetData, '/add_data')

if __name__ == "__main__":
    app.run(debug=True)