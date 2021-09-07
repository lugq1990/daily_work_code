# -*- coding:utf-8 -*-
"""
To check the files in the bucket or not with service account

@author: Guangqiang.lu
"""
import os
import json

file_name = "ServiceKey_DSDBD_storage.json"

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = file_name

with open(file_name, 'r') as f:
    d = f.read()

data = json.loads(d)

print(data['project_id'])

from google.cloud import storage

bucket_name ="npd-65343-datalake-bd-11811-dsd-npd-bd-ca-dsdgcs-raw"

client = storage.Client(data['project_id'])

bucket = client.get_bucket(bucket_name)

# This is to check the file exist or not
file_name = "AgentConnectionDetail_brazil.csv"

blob = bucket.blob(file_name)
print(blob.exists())

# to get whole blobs
blobs = list(bucket.list_blobs())

# try to download whole blob files into local server
file_name_list = []
for blob in blobs:
    print("get file:{} ".format(blob.name))
    blob.download_to_filename(blob.name)
    file_name_list.append(blob.name)


for file in file_name_list:
    with open(file, 'r') as f:
        d = f.readlines()
    print("File: {} has {} records".format(file, len(d)))


# # let's just download this file into local disk to test
# blob.download_to_filename(file_name)
#
# # re-load into memory
# import pandas as pd
#
# df = pd.read_csv(file_name)
