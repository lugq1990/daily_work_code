"""How to get Dataframe from BQ and JSON string."""
import pandas as pd
from google.cloud import bigquery


class BQDataframe(object):
    def __init__(self):
        self.client = bigquery.Client()

    def get(self, query):
        """Must add with try catch, so if we get error, couldn't raise it otherwise later step won't run.

        Raises:
            RuntimeError: [description]

        Returns:
            [type]: [description]
        """
        try:
            query = self.client.query(query)
            _ = query.result()

            df_bq = query.to_dataframe()
        except Exception as e:
            return None

        return df_bq


class JSONDataframe:
    def get(self, json_obj):
        try:
            df = pd.DataFrame(json_obj)

            return df
        except ValueError as e:
            raise ValueError(
                "Please check Pubsub message's data, when try to get DataFrame from JSON get error: {}".format(
                    e
                )
            )
