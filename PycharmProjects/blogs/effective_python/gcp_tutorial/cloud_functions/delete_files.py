from typing import type_check_only
from google.cloud import storage
import time


bucket_name = "cloud_sch_test"
to_delete_file_list = ["iris.zip"]


client = storage.Client()


def hello_gcs(event, context):
    """Triggered by a change to a Cloud Storage bucket.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    upload_file_name = event['name']
    
    print("Get upload file name: {} from event.".format(upload_file_name))

    delete_files()


def delete_files():
    # first try to get file list
    bucket = client.get_bucket(bucket_name)
    # file_list = [blob.name for blob in list(bucket.list_blobs())]

    if len(to_delete_file_list) != 0:
        for file_name in to_delete_file_list:
            try:
                blob = bucket.get_blob(file_name)
                if blob is None:
                    continue
                blob.delete()
                print("GOOD NEWS: File: {} has been deleted".format(file_name))
            except Exception as e:
                print("When try to delete file: {} get error:{}".format(file_name, e))
    else:
        pass

    print("End of cloud functions.")
