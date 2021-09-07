from flask import Flask,  request
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
from flask import jsonify

db_connection = create_engine("sqlite:///test.db")

app = Flask(__name__)
api = Api(app)

conn = db_connection.connect()
conn.execute("Create table if not exists employees(name string, id int)")

# conn.execute("insert into employees (name, id) values ('lu', 1)")

# res = conn.execute("select * from employees")
# print(res.cursor.fetchall())

class Employess(Resource):
    def get(self):
        conn = db_connection.connect()
        query = conn.execute("select * from employees")
        return {"employees": [x[0] for x in query.cursor.fetchall()]}


class EmployeeName(Resource):
    def get(self, id):
        conn = db_connection.connect()
        query = conn.execute("select * from employees where id = {} limit 1".format(id))
        # get only first row
        res = query.first()
        data_dict = {"user_name": res[0], "id": res[1]}
        result = {"data": data_dict}
        return jsonify(result)


class Home(Resource):
    def get(self):
        return jsonify({"data": "hi"})


class HiPost(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        id = int(json_data["id"])
        user_name = str(json_data["user_name"])
        # must create a new thread to do
        conn = db_connection.connect()

        conn.execute("insert into employees (name, id) values ('{}', {})".format(user_name, id))

        result = {"Status": "Successfully"}

        return jsonify(result)


api.add_resource(Employess, '/employees')
api.add_resource(EmployeeName, '/employees/<id>')
api.add_resource(Home, '/')
api.add_resource(HiPost, '/add')


if __name__ == '__main__':
    app.run(port='5002', debug=True)
