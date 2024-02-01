import time
import socket
import random

def send_data():
    host = 'localhost'
    port= 9998
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    s.bind((host, port))
    s.listen()
    print("listen")
    
    client, attr = s.accept()
    
    while True:
        timestamp = int(time.time() * 1000)
        data = f"{timestamp},{random.randint(1, 10)}\n"
        client.send(data.encode())
        print('send: {data}')
        time.sleep(10)
    client.close()
    

if __name__ == '__main__':
    send_data()