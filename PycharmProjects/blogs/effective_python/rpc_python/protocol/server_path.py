
# from xmlrpc.server import SimpleXMLRPCServer


# def is_even(x):
#     print("Get data to proces: {}".format(x))
#     return x % 2 == 0


# port = 5000
# server = SimpleXMLRPCServer(("localhost", port))

# print("Listening on port {}".format(port))
# server.register_function(is_even, 'is_even')
# server.serve_forever()
import sys
import datetime
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler

class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_path = ("/RPC2", )


class ExampleService:
    def get_data(self):
        return 10
    
    class current_time():
        @staticmethod
        def get_current_time():
            return datetime.datetime.now()


ip = "10.237.189.133"
port = 5000

with SimpleXMLRPCServer((ip, port), requestHandler=RequestHandler) as server:
    print("Start to init server side.")
    server.register_introspection_functions()

    server.register_function(pow)

    # @server.register_function
    def add_func(x, y):
        return x + y

    server.register_function(add_func, 'add')

    class MyFunc:
        __name__ = 'mul'
        def mul(self, x, y):
            return x * y
    
    server.register_instance(MyFunc())

    server.register_instance(ExampleService(), allow_dotted_names=True)

    server.register_multicall_functions()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Get key KeyboardInterrupt")
        sys.exit(0)

