# -*- coding:utf-8 -*-
import boto3
import pandas as pd
import paramiko
import time
import os

access_key = 'AKIAJGK5GH7CU46OSS5Q'
secret_key = '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'
bucket_name = 'aliceportal-30899-prod'
already_down_path = '/anaconda-efs/sharedfiles/projects/alice_30899/data_tmp'
path = '/anaconda-efs/sharedfiles/projects/alice_30899/data'
meta_path = 'file_list'
meta_name = 'files.csv'

def execute_command(ssh, command):
    print('Now is %s'%(command))
    stdin, stdout, stderr = ssh.exec_command(command)
    return stdin, stdout, stderr

host = '10.5.105.51'
port = 22
username = 'ngap.app.alice'
password = 'QWer@#2019'
sftp_path = '/sftp/cio.alice'
hdfs_path = '/data/raw/cio/alice/test'

# ssh = paramiko.SSHClient()
# ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# ssh.connect(host, port=22, username=username, password=password)

# command = """scp %s/*.txt guangqiang.lu@10.5.105.51:/sftp/cio.alice/newer/ && rm -f *.txt"""%(path, path)

### get s3 files
s3 = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key).resource('s3')
my_bucket = s3.Bucket(bucket_name)

i = 0
file_list = []
for file in my_bucket.objects.all():
    if file.key.endswith('.txt'):
        i += 1
        file_list.append(file.key)
    if i % 20000 == 0:
        print('already get %d' % (i))

## dump data result to disk
# df = pd.DataFrame(file_list)
# df.columns = ['file']
# df.to_csv(os.path.join(os.path.join(path, meta_path), meta_name))

### HERE just to load file from local server
print('Now is get original data: ')
file_org_tmp = os.listdir(path)
file_org = ['TextFiles/'+ x for x in file_org_tmp if '.txt' in x]
# # store already download file names to disk
#
# df = pd.DataFrame(file_org)
# df.columns = ['files']
# df.to_csv(os.path.join(os.path.join(path, meta_path), "org.csv"))

print('Now is loading all files')
file_list_new = pd.read_csv(os.path.join(os.path.join(already_down_path, meta_path), meta_name))
file_list_new = file_list_new.file.values.tolist()
file_list = list(set(file_list_new) - set(file_org))
file_list = file_list_new


# download files
start_time = time.time()
# path_others = "/anaconda-efs/sharedfiles/projects/alice_30899/data_tmp"
path_mml = "/anaconda-efs/sharedfiles/projects/alice_30899/data"

folder_list = ['data_0',  'data_1',  'data_2',  'data_3',  'data_4',  'data_5',  'data_6',  'data_7',  'data_8',  'data_9',
               'data_A',  'data_B',  'data_C',  'data_D',  'data_E',  'data_F',  'data_G',  'data_H',
               'data_I',  'data_J',  'data_K',  'data_L',  'data_M',  'data_N',  'data_O',  'data_P',
               'data_Q',  'data_R',  'data_S',  'data_T',  'data_U',  'data_V',  'data_W',  'data_X',
               'data_Y',  'data_Z']

end_flag = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
            'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("dest", type=str, help='which folder to download')

args = parser.parse_args()
dest_folder_index = args.dest.upper()

j = 0
for i in range(len(file_list)):
    # Get the file end charac
    try:
        flag = file_list_new[i].split('/')[-1][-5].upper()
    except:
        flag = None
    if flag == dest_folder_index:
        j += 1
        des_folder = folder_list[end_flag.index(flag)]
        # print('Now to process %s' % (des_folder))
        # print('file: ', file_list[i].split('/')[-1])
        my_bucket.download_file(file_list[i], os.path.join(os.path.join(path_mml, des_folder), file_list[i].split('/')[-1]))
    if flag is None:
        my_bucket.download_file(file_list[i],
                                os.path.join(os.path.join(path_mml, "others"), file_list[i].split('/')[-1]))
    if j % 100 == 0:
        print('Now has exported %d'%(j))
        print('Now has used %.2f minutes'%((time.time() - start_time)/60))



# num_list = []
# for f in folder_list:
#     num_list.append(len(os.listdir(os.path.join(path_mml, f))))
# already_get_files = []
# for folder in folder_list:
#     curr_file = os.listdir(os.path.join(path_mml, folder))
#     already_get_files.extend(curr_file)
# out_list = []
# files = os.listdir(os.path.join(path_mml, folder_list[0]))
# i = 0
# for f in files:
#     i += 1
#     if i > 20:
#         break


"""Here is used to get the all not updated files by reading files in HDFS"""
from hdfs3 import HDFileSystem
hdfs = HDFileSystem(host='10.5.105.51', port=8020)
hdfs_path = '/data/insight/cio/alice/contract_all'
result_list = hdfs.list(hdfs_path)




import boto3
import pandas as pd
import os
access_key = 'AKIAJGK5GH7CU46OSS5Q'
secret_key = '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'
bucket_name = 'aliceportal-30899-prod'
path = '....'

s3 = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key).resource('s3')
my_bucket = s3.Bucket(bucket_name)
file_list = []
for f in my_bucket.objects.all():
    if f.key.endswith('.txt'):
        file_list.append(f.key[10:])
df = pd.DataFrame(file_list)
df.to_excel(os.path.join(path, 's3_file_name.xlx'))

# Get all file names, this is on production server:
# import subprocess
# res = subprocess.Popen(['hadoop', 'fs', '-ls', '/data/insight/cio/alice/contract_all'], stdout=subprocess.PIPE)
# hdfs_file_list = []
# for line in res.stdout:
#     hdfs_file_list.append(line)

### This is MML server
from hdfs.ext.kerberos import KerberosClient
client = KerberosClient("http://name-node.cioprd.local:50070")
hdfs_file_list = client.list("/data/insight/cio/alice/contract_all")
# file_name_list = [x[:-4] for x in hdfs_file_list]

# Because for now file_list_new are also combined with the directory folder name, drop it
file_already_list = [x[10:] for x in file_list_new]
not_satified_list = list(set(file_already_list) - set(hdfs_file_list))




"""This is used to make the hdfs file combined with sql files"""
from pyspark.sql import SparkSession
sql_command = "select documentname from alice_uat_staging_30899.es_metadata_ip where (upper(DocumentLanguage)='ENGLISH' or upper(DocumentLanguage)='N/A')"
spark = SparkSession.builder.getOrCreate()
# Here is HIVE dataFrame
df = spark.sql(sql_command)

# HERE is HDFS dataFrame
from hdfs.ext.kerberos import KerberosClient
import pandas as pd
import os
client = KerberosClient("http://name-node.cioprd.local:50070")

hdfs_file_list = client.list("/data/insight/cio/alice/contract_all")
file_name_list = [x[:-4] for x in hdfs_file_list]
df_new = pd.DataFrame(file_name_list)
df_new.name = ['documentname']
df_new = spark.createDataFrame(df_new)

# Here now that I have get the two DataFrames, create template table
df.createTempView('tmp1')
df_new.createTempView('tmp2')
# Here is to join the two tables
df_combine = spark.sql('select t1.documentname from tmp1 t1, tmp2 t2 where t1.documentname = t2.documentname')
df_combine = df_combine.toPandas()

# Store the DataFrame to local disk
path = '...'
df_combine.to_csv(os.path.join(path, 'hive_hdfs_english.csv'))




"""bellow is used to get the file size of the already downlaod files"""
import os
path = "/anaconda-efs/sharedfiles/projects/alice_30899/data"

folder_list = ['data_0',  'data_1',  'data_2',  'data_3',  'data_4',  'data_5',  'data_6',  'data_7',  'data_8',  'data_9',
               'data_A',  'data_B',  'data_C',  'data_D',  'data_E',  'data_F',  'data_G',  'data_H',
               'data_I',  'data_J',  'data_K',  'data_L',  'data_M',  'data_N',  'data_O',  'data_P',
               'data_Q',  'data_R',  'data_S',  'data_T',  'data_U',  'data_V',  'data_W',  'data_X',
               'data_Y',  'data_Z']
# loop for each folder
file_dict = dict()
for f in folder_list:
    print('Now is folder: %s'%(f))
    files = os.listdir(os.path.join(path, f))
    for single_file in files:
        file_dict[single_file] = os.path.getsize(os.path.join(os.path.join(path, f), single_file))
        # file_dict[single_file] = str(int(os.path.getsize(os.path.join(os.path.join(path, f), single_file)) / 1024)) + 'k'

# convert the dictory to dataframe
s = pd.Series(file_dict, name='file_size')
s.index.name = 'file_name'
df = s.reset_index()
df.to_csv(os.path.join(path, 'all_file_size_with_bytes.csv'), index=False)



"""This code is used to upload file to S3 bucket"""
import boto3
import numpy as np
import random
import os

access_key = 'AKIAJGK5GH7CU46OSS5Q'
secret_key = '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'
bucket_name = 'aliceportal-30899-prod'
s3_folder_name = 'Delta'

path = "/anaconda-efs/sharedfiles/projects/alice_30899/data"

folder_list = ['data_0',  'data_1',  'data_2',  'data_3',  'data_4',  'data_5',  'data_6',  'data_7',  'data_8',  'data_9',
               'data_A',  'data_B',  'data_C',  'data_D',  'data_E',  'data_F',  'data_G',  'data_H',
               'data_I',  'data_J',  'data_K',  'data_L',  'data_M',  'data_N',  'data_O',  'data_P',
               'data_Q',  'data_R',  'data_S',  'data_T',  'data_U',  'data_V',  'data_W',  'data_X',
               'data_Y',  'data_Z']

s3 = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key).resource('s3')
my_bucket = s3.Bucket(bucket_name)


# Here is to get some sample data from the first folder of SFTP
k = 20000
file_list = os.listdir(os.path.join(path, folder_list[0]))
rand_list = random.sample(np.arange(len(file_list)).tolist(), k)
rand_file_list = [file_list[x] for x in rand_list]

# Then here is just to upload file to the S3 bucket folder
for i in range(len(rand_file_list)):
    my_bucket.upload_file(os.path.join(os.path.join(path, folder_list[0]), rand_file_list[i]), s3_folder_name + '/' + rand_file_list[i])
    if i % 100 == 0:
        print('Now has uploaded %d'%(i))


"""This is used to delete the file with zero bytes """
# Here because I want to ensure all the files in the new making s3 bucket, I will drop all the files that are just 0 bytes
# so first is to get all the files in the s3 bucket
s3_files = []
i = 0
for f in my_bucket.objects.filter(Prefix=s3_folder_name + '/'):
    if f.key.endswith('.txt'):
        s3_files.append(f.key)
        i += 1
    if i % 2000 == 0:
        print('Now get %d files'%(i))
# because the s3 file list with the folder name, so I just remove the folder name from the list
s3_files = [m[6:] for m in s3_files]
# after get all the files in the s3, now I have get the files in s3 file size, store the file with 0 bytes
# here because I know which folder file that I upload to the S3, so here is just loop for all the file in s3
zero_file_list = []
for f in s3_files:
    if os.path.getsize(os.path.join(os.path.join(path, folder_list[0]), f)) == 0:
        zero_file_list.append(f)
    if f == s3_files[-1]:
        print('zero files get finished')
# after get all the zero bytes files, here I want to make the zero file list with the s3 bucket file file
# and use my_bucket to delete the file from s3
should_delete_file_list = [s3_folder_name + '/' + x for x in zero_file_list]
# loop for the file that should be deleted
for f in should_delete_file_list:
    s3.Object(bucket_name, f).delete()





"""bellow functions is used to make the new file in the S3 folder to make for the s3 source part"""
import boto3
import os
import json
import datetime

access_key = 'AKIAJGK5GH7CU46OSS5Q'
secret_key = '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'
bucket_name = 'aliceportal-30899-prod'
s3_folder_name = 'Delta'
s3 = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key).resource('s3')
my_bucket = s3.Bucket(bucket_name)

# First make one json file to the disk for later step to uplaod files, this folder needs to be configured in the server side.
file_path = '.'

# After make the date time file step, then make a new file name with the current date
# Here I don't want to rename, because there maybe some file in the folder, also I don't know previous date file name
# so here just to make another file with current date, and remove the other file in the folder with extension .json files
# first step is to remove all files with '.json' files to ensure just one json file
for f in os.listdir(file_path):
    if f.endswith('.json'):
        os.remove(f)

# second, we could make another json file with current date, here now date should be UTC datetime
curr_date_file_name = datetime.datetime.utcnow().strftime('%Y_%m_%d') + '.json'
with open(os.path.join(file_path, curr_date_file_name), 'w') as f:
    json.dump(curr_date_file_name, f)
# third step, we delete all .json file in s3
delete_files_list = []
for f in my_bucket.objects.filter(Prefix=s3_folder_name+'/'):
    if f.key.endswith('.json'):
        delete_files_list.append(f.key)
# delete file with loop
[s3.Object(bucket_name, f).delete() for f in delete_files_list]

# fourth step to upload the file with current date, this means just one json file in the s3 with current date
my_bucket.upload_file(os.path.join(file_path, curr_date_file_name), s3_folder_name + '/'+ curr_date_file_name)





"""This function is used to download files from HDFS to local directory"""
from hdfs.ext.kerberos import KerberosClient
import boto3
import numpy as np
import random
import os
import shutil

client = KerberosClient("http://name-node.cioprd.local:50070")

hdfs_path = '/data/insight/cio/alice/contract_all'
local_file_path = '/anaconda-efs/sharedfiles/projects/alice_30899/hdfs_test_data'
parent_folder = '/anaconda-efs/sharedfiles/projects/alice_30899'

# these folders are used to get the satisfied files both in HDFS and S3 bucket, should be same for comparing
local_hdfs_path = os.path.join(parent_folder, 'hdfs_files')
local_s3_path = os.path.join(parent_folder, 's3_files')

# This is the whole S3 bucket files folder
# cause here should be the whole files in the s3 bucket that should be downloaded from HDFS
s3_whole_path = 's3_whole_files'
hdfs_info_folder = 'hdfs_info_folder'
if not os.path.exists(s3_whole_path):
    os.mkdir(os.path.join(parent_folder, s3_whole_path))
if not os.path.exists(hdfs_info_folder):
    os.mkdir(os.path.join(parent_folder, hdfs_info_folder))

access_key = 'AKIAJGK5GH7CU46OSS5Q'
secret_key = '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'
bucket_name = 'aliceportal-30899-prod'
s3_folder_name = 'Delta'
s3 = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key).resource('s3')
my_bucket = s3.Bucket(bucket_name)

# first list all about 40000 files in this folder, also file should length with 15 like this:'0F2U1CUX0UD.txt'
# local_file_path = '/anaconda-efs/sharedfiles/projects/alice_30899/hdfs_test_data'
# all_hdfs_files = os.listdir(local_file_path)
# sati_hdfs_files = [x for x in all_hdfs_files if len(x) == 15]

# n_sample = 10000
# # Here I also random sample these files
# rand_index = random.sample(np.arange(len(sati_hdfs_files)).tolist(), n_sample)
# should_download_file_list = [sati_hdfs_files[i] for i in rand_index]
# # should_download_file_list means the files in the original that should be copied to the hdfs_path,
# # not move but with copy command to keep the original file
# for j in range(len(should_download_file_list)):
#     shutil.copy(os.path.join(local_file_path, should_download_file_list[j]), os.path.join(hdfs_path, should_download_file_list[j]))
#     if j % 1000 == 0:
#         print('Already copied %d files'%(j))

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
    if i % 200 == 0:
        print('Already download %d files'%(i))

# [my_bucket.download_file(f, os.path.join(s3_path, f[6:])) for f in file_list]
# this is for hdfs
sati_file_list = [x[6:] for x in file_list if len(x[6:]) == 15]

# Here is to use client to download whole hdfs files to local directory with the satisfied files
for i in range(len(sati_file_list)):
    client.download(os.path.join(hdfs_path, sati_file_list[i]), os.path.join(local_hdfs_path, sati_file_list[i]))
    if i % 10000 == 0:
        print('Already downlaod %d files'%(i))




"""This is used to copy from one folder to another folder in s3 bucket"""
import boto3

access_key = 'AKIAJGK5GH7CU46OSS5Q'
secret_key = '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'
bucket_name = 'aliceportal-30899-prod'
src_folder_name = 'Delta'
des_folder_name = 'DocStore'
s3 = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key).resource('s3')
my_bucket = s3.Bucket(bucket_name)

# first to get whole files in s3 bucket
file_list = []
for f in my_bucket.objects.filter(Prefix=src_folder_name + '/'):
    if f.key.endswith('.txt'):
        file_list.append(f.key)
# use this code to copy files from one folder to another folder, change the desc file prefix
des_file_list = [x.replace(src_folder_name, des_folder_name) for x in file_list]

for i in range(len(file_list)):
    src_dirc = {'Bucket': bucket_name, 'Key': file_list[i]}
    s3.meta.client.copy(src_dirc, bucket_name, des_file_list[i])
    if i % 100 == 0:
        print('Already copied %d files'%(i))

# Then here is to remove the files in the source part, for now shouldn't used as just comment
# for f in file_list:
#     s3.Object(bucket_name, f).delete()



"""This function is used to make the file upload from local to S3 for testing, just with normal files,
cause if for here I just make another folder for testing, so that I have to make another EBI job, 
so for here, just remove the original Delta folder"""
import boto3
import numpy as np
import random
import os

access_key = 'AKIAJGK5GH7CU46OSS5Q'
secret_key = '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'
bucket_name = 'aliceportal-30899-prod'
s3_folder_name_test = 'Delta'
s3_folder_name_test = 'DeltaTest'

s3 = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key).resource('s3')
my_bucket = s3.Bucket(bucket_name)

path = "/anaconda-efs/sharedfiles/projects/alice_30899/data"

folder_list = ['data_0',  'data_1',  'data_2',  'data_3',  'data_4',  'data_5',  'data_6',  'data_7',  'data_8',  'data_9',
               'data_A',  'data_B',  'data_C',  'data_D',  'data_E',  'data_F',  'data_G',  'data_H',
               'data_I',  'data_J',  'data_K',  'data_L',  'data_M',  'data_N',  'data_O',  'data_P',
               'data_Q',  'data_R',  'data_S',  'data_T',  'data_U',  'data_V',  'data_W',  'data_X',
               'data_Y',  'data_Z']

file_list = os.listdir(os.path.join(path, folder_list[0]))
file_list = [x for x in file_list if len(x) == 15][:15000]
# before upload, here I also remove the whole files in the s3 test folder
[s3.Object(bucket_name, f.key).delete() for f in my_bucket.objects.filter(Prefix=s3_folder_name_test) if f.key.endswith('txt')]

# upload files to s3
upload_files = [s3_folder_name_test + '/' + x for x in file_list]
for i in range(len(file_list)):
    my_bucket.upload_file(os.path.join(os.path.join(path, folder_list[0]), file_list[i]), upload_files[i])
    if i % 1000 == 0:
        print('Already uploaded %d files'%(i))




# """This is used to get the same files in the s3 and hdfs, just make sure that the both folder should be same"""
# import os
#
# parent_folder = '/anaconda-efs/sharedfiles/projects/alice_30899'
# s3_folder = 's3_files'
# hdfs_folder = 'hdfs_files'
# s3_list = os.listdir(os.path.join(parent_folder, s3_folder))
# hdfs_list = os.listdir(os.path.join(parent_folder, hdfs_folder))
#
# commen_list = list(set(s3_list) - set(hdfs_list))
#
# # remove the files that are not in the common list
# # for S3
# for f in s3_list:
#     if f not in commen_list:
#         os.remove(os.path.join(os.path.join(parent_folder, s3_folder), f))
# # for HDFS
# for f in hdfs_list:
#     if f not in commen_list:
#         os.remove(os.path.join(os.path.join(parent_folder, hdfs_folder), f))



##### This is used to download the Delta folder file for testing, there is just about 500 files
import boto3
import numpy as np
import random
import os

access_key = 'AKIAJGK5GH7CU46OSS5Q'
secret_key = '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'
bucket_name = 'aliceportal-30899-prod'
s3_folder_name_test = 'Delta'

s3 = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key).resource('s3')
my_bucket = s3.Bucket(bucket_name)

path = "/anaconda-efs/sharedfiles/projects/alice_30899/data/s3_file_test"

file_list = []
for f in my_bucket.objects.filter(Prefix=s3_folder_name_test+'/'):
    if f.key.endswith('.txt'):
        file_list.append(f.key)

for f in file_list:
    my_bucket.download_file(f, os.path.join(path, f.split('/')[1]))

os.system('scp /anaconda-efs/sharedfiles/projects/alice_30899/data/s3_file_test/*.txt ngap.app.alice@10.5.105.51:/sftp/cio.alice ')


"""This is to test the copy folder from one folder to another folder in S3 bucket"""
#### this is to create a bucket using boto3
import boto3

access_key = 'AKIAJGK5GH7CU46OSS5Q'
secret_key = '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'
bucket_name = 'aliceportal-30899-prod'
session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
client = session.client('s3')
s3 = session.resource('s3')
my_bucket = s3.Bucket(bucket_name)
src_folder_name = 'Delta'
des_folder_name = 'Delta2'
# response = client.put_object(Bucket=bucket_name, Body='', Key=des_folder_name+'/')

file_list = []
for f in my_bucket.objects.filter(Prefix=src_folder_name+'/'):
    if f.key.endswith('.txt'):
        file_list.append(f.key)

des_file_list = [x.replace(src_folder_name, des_folder_name) for x in file_list]

# start copying step
for i in range(len(file_list)):
    src_dic = {'Bucket':bucket_name, 'Key': file_list[i]}
    s3.meta.client.copy(src_dic, bucket_name, des_file_list[i])
    if i % 100 == 0:
        print('Already copy %d files'%(i))

# delete the source folder files in the S3 bucket
[s3.Object(bucket_name, f).delete() for f in file_list]


# this is to remove the whole files with 0 bytes
path = "/anaconda-efs/sharedfiles/projects/alice_30899/data/s3_file_test"
import os
file_list = os.listdir(path)
zero_file_list = []
for f in file_list:
    if os.path.getsize(os.path.join(path, f)) == 0:
        zero_file_list.append(f)
s3_folder = 'Delta'
zero_file_list = [s3_folder + '/' + x for x in zero_file_list]

for f in zero_file_list:
    s3.Object(bucket_name, f).delete()

file_list = []



