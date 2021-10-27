from google.cloud import pubsub_v1
import json
import base64
import os
import pandas as pd

PROJECT_ID = os.environ.get("project")
TOPIC_ID = os.environ.get("publishTopic")

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)

def parse_test(event, context):
    """

    Args:
        event(dict): Event payload.
        context (google.cloud.functions.Context): Metadata for the event.

    Returns:
        identity(str): id of the test case
        query(str): query for BigQuery to get expected result
        data(pandas.DataFrame): actual result, all columns being string
        schema(dict): types and additional parsing argument for features
        mode(str): mode of the test case. Currently should be "compare"

    Example:
        schema = {"col1": {"type": "string"}, "col2": {"type":"float", "digits": 2}}

    """
    message = json.loads(base64.b64decode(event['data']).decode('utf-8'))
    identity = message["id"]
    query = message["query"]
    data = pd.DataFrame(message["data"], dtype=str)
    schema = message["schema"]
    mode = message["mode"]

    return identity, query, data, schema, mode

def construct_message(identity, status, **kwargs):
    """ construct test result output message for Pub/Sub

    Args:
        identity(str): identity number
        status(bool): whether the expected matches actual
        **kwargs:

    Returns:
        message(str): message for Pub/Sub
    """
    dictionary = {"id": identity, "status": status}
    dictionary.update(kwargs)
    message = json.dumps(dictionary)
    return message

def publish(message):
    """ publish test result to Pub/Sub"""
    future = publisher.publish(topic_path, message.encode("utf-8"))

    try:
        print(future.result())
    except:  # noqa
        print("Please handle {} for {}.".format(future.exception(), message))