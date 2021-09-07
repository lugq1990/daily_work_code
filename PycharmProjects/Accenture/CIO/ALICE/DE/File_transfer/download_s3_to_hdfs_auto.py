# -*- coding:utf-8 -*-
import os
import boto3
import datetime
import paramiko
from paramiko import AuthenticationException
import time

host = '10.5.105.51'
port = 22
username = 'ngap.app.alice'
password = 'QWer@#2019'

access_key = 'AKIAJGK5GH7CU46OSS5Q'
secret_key = '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'
bucket_name = 'aliceportal-30899-prod'
folder_name = 'Delta'
mml_daily_s3_folder = '/anaconda-efs/sharedfiles/projects/alice_30899/s3_daily'
prod_sftp_folder = '/sftp/cio.alice/s3_daily'

date_str = datetime.datetime.now().strftime('%Y%m%d')

date_folder = os.path.join(mml_daily_s3_folder, date_str)
try:
    if not os.path.exists(date_folder):
        os.mkdir(date_folder)
except Exception as e:
    pass

s3 = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key).resource('s3')
my_bucket = s3.Bucket(bucket_name)


# first step is to list the whole files in S3 bucket
file_list = []
for f in my_bucket.objects.filter(Prefix=folder_name + '/'):
    if f.key.endswith('.txt'):
        file_list.append(f.key)
    if len(file_list) % 10000 == 0:
        print('Already get %d files'%(len(file_list)))

# after get the file list, then download whole files to the daily folder
[my_bucket.download_file(f, os.path.join(date_folder, f.split('/')[-1])) for f in file_list]

# now that we get the whole files in S3, then I should put the files to the production SFTP folder
# here just use scp command to do that
prod_sftp_folder = os.path.join(prod_sftp_folder, date_str)
try:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username=username, password=password, look_for_keys=False)
except AuthenticationException as e:
    raise AuthenticationException(e)

# make datetime directory
ssh.exec_command('mkdir -p %s'%(prod_sftp_folder))

# in case the authorization code doesn't finish running, so here I want to make the code to wait
time.sleep(1)

# this command runs at MML server, so here must us os to run
command = 'scp %s/*.txt ngap.app.alice@10.5.105.51:%s/'%(date_folder, prod_sftp_folder)
os.system(command)   # I have google that the process with os dose wait to run finishing


# After whole process has finished, then here I just want to put the files in SFTP folder to HDFS
hdfs_path = '/data/raw/cio/alice/'
hdfs_path = os.path.join(hdfs_path, date_str)
put_command = "hdfs dfs -mkdir -p %s && hdfs dfs -put %s/*.txt %s/"%(hdfs_path, prod_sftp_folder, hdfs_path)
ssh.exec_command(put_command)

print('*'*20)
print('Whole step finished!')



### bellow is testing code
import os
import boto3

host = '10.5.105.51'
port = 22
username = 'ngap.app.alice'
password = 'QWer@#2019'

access_key = 'AKIAJGK5GH7CU46OSS5Q'
secret_key = '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'
bucket_name = 'aliceportal-30899-prod'
folder_name = 'Delta'

s3 = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key).resource('s3')
my_bucket = s3.Bucket(bucket_name)

file_list = []
for f in my_bucket.objects.filter(Prefix=folder_name + '/'):
    if f.key.endswith('.txt'):
        file_list.append(f.key)


# here is to copy files in another s3 folder to source folder
src_folder_name = 'Delta2'
des_folder_name = 'Delta'
file_list = []
for f in my_bucket.objects.filter(Prefix=src_folder_name + '/'):
    if f.key.endswith('.txt'):
        if my_bucket.Object(f.key).content_length == 0:
            continue
        file_list.append(f.key)
print('How many files: ',len(file_list))
des_file_list = [x.replace(src_folder_name, des_folder_name) for x in file_list]

for i in range(len(file_list)):
    src_dirc = {'Bucket': bucket_name, 'Key': file_list[i]}
    s3.meta.client.copy(src_dirc, bucket_name, des_file_list[i])
    if i % 100 == 0:
        print('Already copied %d files' % (i))




# here I found that for the Delta folder, I found that for some files are 0 files
size_dict = dict()
for f in file_list:
    size_dict[f] = my_bucket.Object(f).content_length
size_list = list(size_dict.values())
import numpy as np
size_array = np.array(size_list)
np.sum(size_array == 0)
print('Total files %d, the files with 0 bytes file number: %d'%(len(file_list), np.sum(size_array == 0)))

# here to remove the whole 0 files
delete_list = []
for key, v in size_dict.items():
    if v == 0:
        delete_list.append(key)
for f in delete_list:
    my_bucket.Object(f).delete()





