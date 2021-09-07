
import xmlrpc.client as client

ip = "10.237.189.133"
port = 5000
server = client.ServerProxy("http://{}:{}/".format(ip, port))

multi = client.MultiCall(server)

multi.get_data()

multi.pow(2, 9)

multi.add(1,2 )

try:
    print(server.current_time.get_current_time())
except:
    print("get error!")

multi = client.MultiCall(server)

multi.get_data()

multi.pow(2, 9)

multi.add(1,2 )

try:
    for r in multi():
        print(r)
except Exception as e:
    print("get error: ", e)