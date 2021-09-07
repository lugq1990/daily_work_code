import base64

def say_hi(event, context):
    """Triggered from a message on a Cloud Pub/Sub topic.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    print("Functions have been called!")
    pubsub_message = base64.b64decode(event['data']).decode('utf-8')
    print(pubsub_message)