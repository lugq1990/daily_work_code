# -*- coding:utf-8 -*-
"""
This is an example to show how to use python to pubsub data into topics.

@author: Guangqiang.lu
"""
import datetime, json, os, random, time

# Set the `project` variable to a Google Cloud project ID.
project = 'cloudtutorial-279003'

FIRST_NAMES = ['Monet', 'Julia', 'Angelique', 'Stephane', 'Allan', 'Ulrike', 'Vella', 'Melia',
    'Noel', 'Terrence', 'Leigh', 'Rubin', 'Tanja', 'Shirlene', 'Deidre', 'Dorthy', 'Leighann',
    'Mamie', 'Gabriella', 'Tanika', 'Kennith', 'Merilyn', 'Tonda', 'Adolfo', 'Von', 'Agnus',
    'Kieth', 'Lisette', 'Hui', 'Lilliana',]
CITIES = ['Washington', 'Springfield', 'Franklin', 'Greenville', 'Bristol', 'Fairview', 'Salem',
    'Madison', 'Georgetown', 'Arlington', 'Ashland',]
STATES = ['MO','SC','IN','CA','IA','DE','ID','AK','NE','VA','PR','IL','ND','OK','VT','DC','CO','MS',
    'CT','ME','MN','NV','HI','MT','PA','SD','WA','NJ','NC','WV','AL','AR','FL','NM','KY','GA','MA',
    'KS','VI','MI','UT','AZ','WI','RI','NY','TN','OH','TX','AS','MD','OR','MP','LA','WY','GU','NH']
PRODUCTS = ['Product 2', 'Product 2 XL', 'Product 3', 'Product 3 XL', 'Product 4', 'Product 4 XL', 'Product 5',
    'Product 5 XL',]


# Here I use python client to do the pubsub module
from google.cloud import pubsub_v1

topic_name = 'first_python_topic'
# Configure the batch to publish as soon as there is ten messages,
# one kilobyte of data, or one second has passed.
batch_settings = pubsub_v1.types.BatchSettings(
    max_messages=10,  # default 100
    max_bytes=1024,  # default 1 MB
    max_latency=1,  # default 10 ms
)

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project, topic_name)

futures = dict()


# here we have to make the callback from future
# def get_callback(f, data):
#     def callback(f):
#         try:
#             print(f.result)
#             futures.pop()
#         except:
#             print("Please handle {} for {}".format(f.exception(), data))
#
#     return callback
# # let's pub the data
# for i in range(1):
#     first_name, last_name = random.sample(FIRST_NAMES, 2)
#     data = {
#     'tr_time_str': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#     'first_name': first_name,
#     'last_name': last_name,
#     'city': random.choice(CITIES),
#     'state':random.choice(STATES),
#     'product': random.choice(PRODUCTS),
#     'amount': str(float(random.randrange(50000, 70000)) / 100),
#     }
#     data = json.dumps(data)
#     data = str(i)
#     futures.update({data: None})
#     future = publisher.publish(topic_path, data=data.encode('utf-8'))
#     futures[data] = future
#
#     future.add_done_callback(get_callback(future, data))


# in fact, we could just do our pubsub logic into a function that we could use
def callback(future):
    message_id = future.result()
    print(message_id)

def pubsub_data(data):
    if isinstance(data, dict):
        data = json.dumps(data)
    elif not isinstance(data, str):
        raise ValueError("We could only support with str data or JSON data")

    data = data.encode('utf-8')
    try:
        future = publisher.publish(topic_path, data)
        # print("Get item id: {}".format(future.result()))
        # we could even do without blocking, so that we could do batch multiple message
        future.add_done_callback(callback)
    except Exception as e:
        raise Exception("When try to publish data with error: {} for data: {}".format(future.exception(), data))


# let's implement with subscribe logic
# StreamingPull service API to implement the asynchronous client API efficiently
from concurrent.futures import TimeoutError

subscriber = pubsub_v1.SubscriberClient()
sub_name = 'first_sub'
subscr_path = subscriber.subscription_path(project, sub_name)


def callback_sub(message):
    print('Get message:', message)
    message.ack()


streaming_pull_future = subscriber.subscribe(subscr_path, callback=callback_sub)



if __name__ == '__main__':
    n = 4
    for _ in range(n):
        first_name, last_name = random.sample(FIRST_NAMES, 2)
        data = {
            'tr_time_str': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'first_name': first_name,
            'last_name': last_name,
            'city': random.choice(CITIES),
            'state': random.choice(STATES),
            'product': random.choice(PRODUCTS),
            'amount': str(float(random.randrange(50000, 70000)) / 100),
        }
        pubsub_data(data)

    print("We have finished whole thing without error.")

    # get subscribe result
    while True:
        try:
            streaming_pull_future.result(timeout=5.0)
        except TimeoutError:
            streaming_pull_future.cancel()


# for i in range(1):
#     data = u'first data {}'.format(i)
#     first_name, last_name = random.sample(FIRST_NAMES, 2)
#     data = {
#     'tr_time_str': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#     'first_name': first_name,
#     'last_name': last_name,
#     'city': random.choice(CITIES),
#     'state':random.choice(STATES),
#     'product': random.choice(PRODUCTS),
#     'amount': str(float(random.randrange(50000, 70000)) / 100),
#     }
#     data = json.dumps(data)
#     data = data.encode('utf-8')
#
#     future = publisher.publish(topic_path, data)
#     print(future.result())


# this is to use command line to publish data.
# while True:
#   first_name, last_name = random.sample(FIRST_NAMES, 2)
#   data = {
#     'tr_time_str': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#     'first_name': first_name,
#     'last_name': last_name,
#     'city': random.choice(CITIES),
#     'state':random.choice(STATES),
#     'product': random.choice(PRODUCTS),
#     'amount': float(random.randrange(50000, 70000)) / 100,
#   }
#
#   # For a more complete example on how to publish messages in Pub/Sub.
#   #   https://cloud.google.com/pubsub/docs/publisher
#   message = json.dumps(data)
#   command = "gcloud pubsub topics publish transactions --message='{}'".format(message)
#   print(command)
#   os.system(command)
#   time.sleep(random.randrange(1, 5))