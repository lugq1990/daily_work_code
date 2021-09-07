# To make the socket client to connect with server

import socket

host = 'localhost'
port = 12345

# Bind IP and client port
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# start to connect
client_socket.connect((host, port))

data = "hi server!"

# send data to server
client_socket.send(data.encode())

# get result from server
res = client_socket.recv(1024)

print(res.decode())



# # Try with socket client and server socket
# # first with server
# import socket

# host = "10.128.0.17"
# port = 22221

# server = socket.socket()

# server.bind((host, port))
# server.listen()

# while True:
#     c, addr = server.accept()
#     print("Get connection from: {}".format(addr))
#     data = c.recv(1024)
#     print("get data from client: {}".format(data.decode()))
#     c.send("This is server, get your date!".encode())


# # client side

# import socket


# host = '10.128.0.17'
# port = 22221


# client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client.connect((host, port))
# client.send("Hi server.".encode())
# data = client.recv(1024)
# print("Get data from server: {}".format(data.decode()))


# # get real external Ip
# import requests
# url = "https://api.duckduckgo.com/?q=ip&format=json"

# raw = requests.get(url)
# print("real IP: ", raw.json()['Answer'].split()[4])


