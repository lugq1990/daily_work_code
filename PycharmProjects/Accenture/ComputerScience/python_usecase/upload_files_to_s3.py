# -*- coding:utf-8 -*-
"""
This is I write to upload code zip file to S3, so that I could
just download if from server side.

@author: Guangqiang.lu
"""
import boto3
import os
import shutil

path = 'C:/Users/guangqiiang.lu/Documents/lugq/github/'
file_name = 'deployment.zip'

access_key = 'AKIARLSQS4QER3F2RWIB'
secret_key = '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'
bucket_name = '30899-aliceportal-dev'


session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
s3 = boto3.resource('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)

folder_name = 'test'
my_bucket = s3.Bucket(bucket_name)


client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
client.upload_file(os.path.join(path, file_name), bucket_name, folder_name+'/'+file_name)
print("File has been uploaded.")

# this is to download file from s3 to local
# des_path = "C:/Users/guangqiiang.lu/Documents/lugq/github"
# for file in my_bucket.objects.filter(Prefix=folder_name + "/"):
#     if file.key.endswith("token.zip"):
#         print("Now to download:", file.key)
#         my_bucket.download_file(file.key, os.path.join(des_path, file.key.split('/')[-1]))



# this is to upload file to S3 from s3 bucket
# import boto3
# import os
#
# path = '.'
# file_name = "dep.zip"
#
# access_key = 'AKIARLSQS4QER3F2RWIB'
# secret_key = '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'
# bucket_name = '30899-aliceportal-dev'
#
# session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
# s3 = boto3.resource('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
# client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
#
# folder_name = 'test'
# my_bucket = s3.Bucket(bucket_name)
#
# client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
# client.upload_file(os.path.join(path, file_name), bucket_name, folder_name+'/'+file_name)



# import os
# path = '.'
# str_to_replace = "/anaconda-efs/envs/cap_prd_py36_mysched_9376/bin/python"
# str_replace_with = "/anaconda-efs/envs/cap_prd_py36_mysched9376/bin/python"
# file_list = sorted([x for x in os.listdir(path) if x.endswith("sh")])
#
# key_tab = "kinit -k -t /etc/security/keytabs/sa.mmlp.mysched.keytab sa.mmlp.mysched \n"
#
# write_list = ["run_pipeline_01_dicts.py", "run_pipeline_02_tfidf.py", "run_pipeline_03_lsi.py"]
# for i in range(len(write_list)):
#     print("Now to write:", file_list[i])
#     with open(file_list[i], 'w') as f:
#         f.write(key_tab + str_replace_with + " " + write_list[i])
#
# with open(file_list[0]) as f:
#     print(f.readlines())



