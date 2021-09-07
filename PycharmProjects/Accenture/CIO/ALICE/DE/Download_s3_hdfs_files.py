from hdfs.ext.kerberos import KerberosClient
import boto3
import numpy as np
import random
import os
import shutil
import datetime

client = KerberosClient("http://name-node.cioprd.local:50070")

hdfs_path = '/data/insight/cio/alice/contract_all'
local_file_path = '/anaconda-efs/sharedfiles/projects/alice_30899/hdfs_test_data'
parent_folder = '/anaconda-efs/sharedfiles/projects/alice_30899'

# these folders are used to get the satisfied files both in HDFS and S3 bucket, should be same for comparing
local_hdfs_path = os.path.join(parent_folder, 'hdfs_files')
local_s3_path = os.path.join(parent_folder, 's3_files')

# Here cause I want to make the whole step to be automatically running, I add the date folder in both
# hdfs and s3 local folder, in case there isn't folder exits, just create one date folder
date_folder = datetime.datetime.now().strftime('%Y%m%d')
local_hdfs_path = os.path.join(local_hdfs_path, date_folder)
local_s3_path = os.path.join(local_s3_path, date_folder)

if not os.path.exists(local_hdfs_path):
    os.mkdir(local_hdfs_path)
if not os.path.exists(local_s3_path):
    os.mkdir(local_s3_path)


# This is the whole S3 bucket files folde
access_key = 'AKIAJGK5GH7CU46OSS5Q'
secret_key = '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'
bucket_name = 'aliceportal-30899-prod'
s3_folder_name = 'Delta_Duplicate_Test'
s3 = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key).resource('s3')
my_bucket = s3.Bucket(bucket_name)


"""This step is to download file from s3, then downlaod satisfied file from HDFS, if you want to use the 
python code to test for the future Delta logic, please use the second part code for doing that!"""
### fisrt downlaod all files in S3
# after copy to the local disk, I should download the file from S3 bucket with these files
# but before I should add the prefix with the file name to download
# this step couldn't be implement, because here I should first download file from S3, then with HDFS
file_list = []
for f in my_bucket.objects.filter(Prefix=s3_folder_name+'/'):
    if f.key.endswith('.txt'):
        file_list.append(f.key)
# Here is to download s3 files
for i in range(len(file_list)):
    my_bucket.download_file(file_list[i], os.path.join(local_s3_path, file_list[i][6:]))
    if i % 1000 == 0:
        print('Already download %d files'%(i))

# [my_bucket.download_file(f, os.path.join(s3_path, f[6:])) for f in file_list]
# this is for hdfs
sati_file_list = []
for f in file_list:
    if len(f.split('/')[1]) == 15:
        sati_file_list.append(f.split('/')[1])
# sati_file_list = [x[6:] for x in file_list if len(x[6:]) == 15]

# Here is to use client to download whole hdfs files to local directory with the satisfied files
for i in range(len(sati_file_list)):
    client.download(os.path.join(hdfs_path, sati_file_list[i]), os.path.join(local_hdfs_path, sati_file_list[i]))
    if i % 1000 == 0:
        print('Already downlaod %d files'%(i))


"""This is the second part for delta logic, cause there must be some file's name not satified with HDFS put logic,
so here should be first download hdfs file, then download file from s3 with the satisfied files"""
# cause the delta logic with be the date as folder, so here is the time folder
import datetime
curr_date = datetime.datetime.now().strftime('%Y%m%d')
# or you could change as you want
hdfs_parent_folder = '/data/insight/cio/alice/'
hdfs_file_list = client.list(os.path.join(hdfs_parent_folder, curr_date))

# then for the already list files, combined with the s3 bucket prefix folder name
s3_file_list = [s3_folder_name + '/' + f for f in hdfs_file_list]

# this is to download HDFS files
for f in hdfs_file_list:
    client.download(os.path.join(hdfs_path, f), os.path.join(local_hdfs_path, f))

# this is to download S3 bucket
for f in s3_file_list:
    my_bucket.download_file(f, os.path.join(local_s3_path, f.split('/')[1]))
