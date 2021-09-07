# -*- coding:utf-8 -*-
"""
This is for testing with server side

@author: Guangqiang.lu
"""
import xmlrpc.client as client

proxy = client.ServerProxy("http://localhost:8080")

print("Add:", proxy.add(1, 2))
print("Power: ", proxy.pow(2, 3))
print("Mul:", proxy.mul(2, 3))

print("whole func:", proxy.system.listMethods())