
from confluent_kafka import Consumer
from datetime import date, datetime
import os
from hashlib import md5
import json


broker = 'localhost:9092'
topic = 'first'


def get_consumer():
    consumer = Consumer({
        'bootstrap.servers': broker,
        'group.id': 'new-group',
        'auto.offset.reset': 'earliest'
    })

    consumer.subscribe([topic])
    return consumer

consumer = get_consumer()

def open_file(data):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open("tmp.txt", 'a') as f:
        f.write(now + "\t" + str(data) + '\n')


def get_message(n, time_window=1.0):
    msg = consumer.poll(time_window)
    n += 1
    return n, msg


def get_msg_number():
    n = 0
    while True:
        n, msg = get_message(n)
        if msg is None:
            continue
        if msg.error():
            print("Consumer error: {}".format(msg.error()))
            continue
        
        if n % 100 == 0:
            print("Already get {} records.".format(n))
            open_file(n)


def write_data_to_file(data):
    with open('hash_value.txt', 'a') as f:
        f.write(data + '\n')

def hash_msg():
    n = 0
    while True:
        n, msg = get_message(n)
        if msg is None:
            continue
        if msg.error():
            print("Consumer error:{}".format(msg.error))
        
        msg_value = msg.value()
        msg_hash = md5(msg_value).hexdigest()
        
        value_json = {'msg_value': msg_value.decode('utf-8'), 'mgs_md5':msg_hash}
        print(value_json)
        
        value_str = json.dumps(value_json)
        write_data_to_file(value_str)
        print("Now to process {}th msg".format(n))
        

if __name__ == "__main__":
    hash_msg()