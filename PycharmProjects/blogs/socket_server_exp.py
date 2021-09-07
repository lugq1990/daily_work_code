# This is server side
import socket

# Create Server socket client
server_client = socket.socket()

host = 'localhost'
port = 12345

# Which IP and port to bind for server.
server_client.bind((host, port))
# Server will always listen on this port
server_client.listen()


# keep with listening
try:
    while True:
        # If we get request, will also get source address ip
        c, addr = server_client.accept()
        print("Get connection from {}".format(addr))

        data = c.recv(1024)
        print(data.decode())

        # send back to client means that server get the message
        c.send("Hi client, this is server!".encode())
except:
    print("Stop server!")