"""Publish message into pubsub"""

from google.cloud import pubsub_v1


class PublishMessage:
    def __init__(self, project_id=None, topic_id=None):
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_path = self.publisher.topic_path(project_id, topic_id)

    def publish_message(self, message):
        """Publish result message into Pubsub

        Args:
            message ([type]): [description]
            topic_id ([type], optional): [description]. Defaults to None.
            project_id ([type], optional): [description]. Defaults to None.
        """
        res = self.publisher.publish(self.topic_path, message.encode("utf-8"))

        print(res.result())

        # try:
        #     print(res.result())
        # except Exception as e:
        #     print("When try to publish message get error: {}".format(e))
