# -*- coding:utf-8 -*-
"""
RPC is used in client/server architecture, when we need to call
a function implemented in server side, client doesn't need to
know the implement detail of function, we could just call it
from server side, but we have to ensure the server side should
serve the request.

@author: Guangqiang.lu
"""
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler

class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = '/RPC2,'


with SimpleXMLRPCServer(('localhost', 8080), requestHandler=RequestHandler) as server:
    # 内省函数
    server.register_introspection_functions()

    server.register_function(pow)

    def adder(x, y):
        return x + y
    server.register_function(adder, "add")

    class MyFunc:
        def mul(self, x, y):
            return x * y

    server.register_instance(MyFunc())

    server.serve_forever()


