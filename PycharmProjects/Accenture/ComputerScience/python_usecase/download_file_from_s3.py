# -*- coding:utf-8 -*-
"""
This is try to download the file code from s3 to server.

@author: Guangqiang.lu
"""
import boto3
import os
import shutil

local_path = "C:/Users/guangqiiang.lu/Documents/lugq/github/new_hr_model_tuning"

local_path = '/anaconda-efs/sharedfiles/projects/mysched_9376/kt_code/model_training'

access_key = 'AKIARLSQS4QER3F2RWIB'
secret_key = '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'
bucket_name = '30899-aliceportal-dev'

file_name = "deployment_integration.zip"

session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
s3 = boto3.resource('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)

folder_name = 'test'
my_bucket = s3.Bucket(bucket_name)


## before we download, we should remove the folder and file with mysch
for file in os.listdir(local_path):
    print("Start to remove:", file)
    if os.path.isfile(os.path.join(local_path, file)):
        os.remove(file)
    else:
        shutil.rmtree(os.path.join(local_path, file))

for file in my_bucket.objects.filter(Prefix=folder_name + '/'):
    if file.key.endswith(file_name):
        print("Get file:", file.key)
        my_bucket.download_file(file.key, os.path.join(local_path, file.key.split('/')[-1]))







