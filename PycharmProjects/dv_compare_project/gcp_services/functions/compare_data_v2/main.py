"""Data comparation based on BigQuery data and Pubsub message data.


"""
import os
import json
import numpy as np
import pandas as pd
import warnings
import base64
import traceback

from utils.get_data import BQDataframe,  JSONDataframe
from utils.convert_df_util import ConvertDataFrames
from utils.compare import CompareDataframes
from utils.pubsub_util import PublishMessage

warnings.simplefilter('ignore')


def data_comparation(event, context):
    """Main entry point.

    # TODO: There should be a failure handler in real production.

    Args:
        event ([type]): [description]
        context ([type]): [description]
    """
    # project_id = "sbx-65343-autotest8-b-a80bc33f"
    # topic_id = "dv_test_data_result"

    project_id = os.environ.get("project")
    topic_id = os.environ.get("publishTopic")

    # Here we than we could just try to call main compare logic
    # Add logic to avoid data extract from PUBSUB error and output message into pubsub
    identity = ""
    try:
        message = json.loads(base64.b64decode(event['data']).decode('utf-8'))

        identity = message["id"]
        query = message["query"]
        data = message["data"]
        mode = message["mode"]
    except Exception as e:
        if not identity:
            identity = "None"

        error_message = "Retrieve data from PUBSUB with error: " + str(e)
        print(error_message)
        trace_message = traceback.format_exc()

        # add trace message into log
        print(trace_message)

        message = json.dumps({"id": identity, "status":False, 
                "missed_expected":[], "missed_actual":[], "fail_reason": error_message, "trace_message": trace_message})
    
        # No need for other step, just pass
        PublishMessage(project_id, topic_id).publish_message(message)
        return

    try:
        df_bq = BQDataframe().get(query)
        # Add a logic that if we get None from BQ, then just stop and send message into topic.
        if df_bq is None:
            raise RuntimeError("Couldn't get BQ dataframe with SQL: {}, please check SQL!".format(query))

        df_json = JSONDataframe().get(data)

        # Change JSON dataframe columns to same as BQ, so no need for column name failure
        df_json.columns = df_bq.columns

        df_json, df_bq = ConvertDataFrames().convert(df_json, df_bq)
        
        # added with compare result with index and records
        # noted: diff rocords will be dict.
        status, diff_json, diff_bq, diff_index_json, diff_index_bq = CompareDataframes().compare(df_json, df_bq)
        print("After full process, get comparation result: ", status)

        message = json.dumps({"id": identity, "status":status, "missed_expected":diff_json, "missed_actual":diff_bq, "diff_index_json": diff_index_json, "diff_index_bq":diff_index_bq})
    except Exception as e:
        # If there is any kind of error, then `status` will be exception values and will be the first
        # exception will be raised. Must be JSON serialize
        # Also won't get diff list!
        error_message = "Compare fail for reason: {}".format(str(e))
        print(error_message)
        trace_message = traceback.format_exc()

        # add trace message into log
        print(trace_message)

        message = json.dumps({"id": identity, "status":False, 
                "missed_expected":[], "missed_actual":[], "fail_reason": error_message, "trace_message": trace_message})

    # This should be ensure that could be a handler to send out message.
    PublishMessage(project_id, topic_id).publish_message(message)

    print("Full processing finished :)")
