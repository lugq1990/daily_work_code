# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""
import boto3
import os
import shutil
from botocore.client import Config

config = Config(connect_timeout=50, retries={'max_attempts': 0})
# s3 = boto3.client('s3', config=config)

local_path = "C:/Users/guangqiiang.lu/Documents/lugq/github/new_hr_model_tuning/deployment_integration/resources/spacy_pretrained_model"

access_key = 'AKIARLSQS4QER3F2RWIB'
secret_key = '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'
bucket_name = '30899-aliceportal-dev'

file_name = "d.zip"

session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
s3 = boto3.resource('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key, config=config)
client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key, config=config)

folder_name = 'test'
my_bucket = s3.Bucket(bucket_name)


## this is for uploading logic
for file in os.listdir(local_path):
    if file.endswith(file_name):
        print("Get file:", file)
        # my_bucket.upload_file(os.path.join(local_path, file), os.path.join(folder_name, file))
        client.upload_file(os.path.join(local_path, file), bucket_name, folder_name + '/' + file)
        print("Upload finished.")
