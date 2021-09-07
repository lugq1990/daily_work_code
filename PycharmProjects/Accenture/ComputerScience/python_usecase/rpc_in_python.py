# -*- coding:utf-8 -*-
"""RPC is means Remote Procedure Call, used in client-server use case.
Here I just make some examples with server and client side logic.
"""

# SERVER side
# server side to execute the computation
from xmlrpc.server import SimpleXMLRPCServer
import datetime
import xmlrpc.client
import tempfile
import os
import sys


# def is_even(x):
#     return x % 2 == 0

# -------
# process date data
# -------

# def today():
#     today = datetime.datetime.today()
#     return xmlrpc.client.DateTime(today)


#---------
# read files and write filew with RPC
#--------
# tmp_path = tempfile.mkdtemp()
#
# with open(os.path.join(tmp_path, 'image.png'), 'wb') as f:
#     f.write(b"somethind in byte")
#
#
# def python_read():
#     with open(os.path.join(tmp_path, 'image.png'), 'rb') as fr:
#         return xmlrpc.client.Binary(fr.read())
#
#
# # we could even call different function same time
# def divide(x, y):
#     return x / y

# server = SimpleXMLRPCServer(('localhost', 5000))
# print('listening to 5000 port')
# server.register_function(python_read, 'python_read')
# server.register_function(divide, 'divide')
# server.serve_forever()

# -------
# we could do more with server side
# -------

# from xmlrpc.server import SimpleXMLRPCServer
# from xmlrpc.server import SimpleXMLRPCRequestHandler
#
#
# # we could restrict to that path
# class RequestHandler(SimpleXMLRPCRequestHandler):
#     rpc_paths = ('/RPC2',)
#
#
# # create server
# with SimpleXMLRPCServer(('localhost', 5000), requestHandler=RequestHandler) as server:
#     print("look with port 5000")
#     server.register_introspection_functions()
#
#     # with default function
#     server.register_function(pow)
#
#     # we could create another function
#     def add(x, y):
#         return x + y
#     server.register_function(add, 'add')
#
#     # we could even register with an instance
#     class MyFunc:
#         def mul(self, x, y):
#             return x * y
#     server.register_instance(MyFunc())
#
#     server.serve_forever()


#--------
# we could register our server function with @
#--------

from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler


class RequestHander(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2', )


class ExampleService:
    @staticmethod
    def get_data(self):
        return 'test'

    class CurrentTime:
        @staticmethod
        def gct():
            return datetime.datetime.now()


with SimpleXMLRPCServer(('localhost', 5000), requestHandler=RequestHander) as server:
    server.register_introspection_functions()

    print("server seek 5000.")

    server.register_function(pow)

    # with just @ to register
    @server.register_function
    def add_f(x, y):
        return x + y

    server.register_instance(ExampleService(), allow_dotted_names=True)

    server.register_multicall_functions()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("user calls stop")
        sys.exit(0)





"""This should be client side to make the call to server."""
# import xmlrpc.client
#
# with xmlrpc.client.ServerProxy("http://localhost:5000/") as proxy:
#     print("3 is even: %s" % (proxy.is_even(3)))
#     print("10 is even: %s" % (proxy.is_even(10)))


from sklearn.metrics import hinge_loss