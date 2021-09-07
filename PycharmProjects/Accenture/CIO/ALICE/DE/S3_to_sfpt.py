# -*- coding:utf-8 -*-
"""This class is used to transfer file from S3 to local then use the command to move file from MMML server to
production SFTP folder"""

import boto3
import os

production = False

access_key = 'AKIAJMR43CLDBRZMVLAA'
secret_key = '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'
bucket_name = '4339-mmr-prod'
# access_key = 'AKIAJR6ATTW6YPNFMHLA'
# secret_key = '30899-User-DEV-20190301190633'

"""For now, the dev env will get the result from server side"""
### dev env
access_key = 'AKIAJLZTHODMFRKKIZIQ'
secret_key = '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'
bucket_name = '30899-aliceportal-dev'
## This is used to get bucket permission info
client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
client.get_bucket_acl(Bucket=bucket_name)

### staging env
access_key = 'AKIAJESPISZ2QBVPQFGQ'
secret_key = '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'
bucket_name = 'aliceportal-30899-stg'


#
# ### Production env
access_key = 'AKIAJGK5GH7CU46OSS5Q'
secret_key = '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'
bucket_name = 'aliceportal-30899-prod'

session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
s3 = boto3.resource('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)

def create_s3(path):
    s3 = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key).resource('s3')
    my_bucket = s3.Bucket(bucket_name)
    i = 0
    for file in my_bucket.objects.all():
        if file.key.endswith('.txt'):
            i += 1
            my_bucket.download_file(file_list[i], os.path.join(path, file_list[i].split('/')[-1]))
        if i % 5000 == 0:
            print('already download %d' % (i))

# s3 = session.resource('s3')
# bucket_name = '4339-mmr-prod'
my_bucket = s3.Bucket(bucket_name)
i = 0
n = 100
file_list = []

path = '..'
for file in my_bucket.objects.all():
    if file.key.endswith('.txt'):
        i += 1
        file_list.append(file.key)
    if i % 10000 == 0:
        print('already get %d' % (i))
my_bucket.download_file(file_list[i], os.path.join(path, file_list[i].split('/')[-1]))


file_list = []
folder_name = 'Delta'
my_bucket = s3.Bucket(bucket_name)
for f in my_bucket.objects.filter(Prefix=folder_name + '/'):
    if f.key.endswith('.txt'):
        file_list.append(f.key)


bucket_name_list = []
for bucket in s3.buckets.all():
    bucket_name_list.append(bucket.name)

for x in bucket_name_list:
    if '30899' in x or 'alice' in x:
        print(x)
print(bucket_name in bucket_name_list)
print(file_list)

# # Download S3 files to local file
#
# # local_path = '/mrsprd_data/Users/guangqiang.lu/tmp_contract'
# local_path = '/anaconda-efs/sharedfiles/projects/alice_30899/data_tmp'
# for i in range(len(file_list)):
#     if not production:
#         my_bucket.download_file(file_list[i], os.path.join(local_path, file_list[i].split('/')[-1].split('.')[0] + '.txt'))
#     else:
#         my_bucket.download_file(file_list[i], os.path.join(local_path, file_list[i].split('/')[-1]))
#     if i % 10 == 0:
#         print('Already download %d files'%(i))
#
# command = 'scp /mrsprd_data/Users/guangqiang.lu/tmp_contract/*.txt guangqiang.lu@10.5.105.51:/sftp/cio.alice/newer/'
# from subprocess import call
# print('Now is executing command: %s'%(command))
# call(command, shell=True)
# print('Finished command!')

from sklearn.model_selection import cross_validate



"""this is used to download the file from S3 to SFTP"""
import boto3
import pandas as pd
import paramiko
import time
import os

access_key = 'AKIAJGK5GH7CU46OSS5Q'
secret_key = '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'
bucket_name = 'aliceportal-30899-prod'
path = '/anaconda-efs/sharedfiles/projects/alice_30899/data_tmp'
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

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, port=22, username=username, password=password)

command = """scp %s/*.txt guangqiang.lu@10.5.105.51:/sftp/cio.alice/newer/ && rm -f *.txt"""%(path, path)


s3 = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key).resource('s3')
my_bucket = s3.Bucket(bucket_name)

i = 0
file_list = []
for file in my_bucket.objects.all():
    if file.key.endswith('.txt'):
        i += 1
        file_list.append(file.key)
    if i % 20000 == 0:
        print('already download %d' % (i))

df = pd.DataFrame(file_list)
df.columns = ['file']
df.to_csv(os.path.join(os.path.join(path, meta_path), meta_name))
# Download file
# overload = False
# while not overload:
#     try:
#         for i in range(len(file_list)):
#             my_bucket.download_file(file_list[i], os.path.join(path, file_list[i].split('/')[-1]))
#             file_list.pop(i)
#     except:
#         execute_command(command)
#         time.sleep(120)


# download files
start_time = time.time()
path_others = "/anaconda-efs/sharedfiles/projects/alice_30899/data_tmp/data_others"
for i in range(len(file_list)):
    my_bucket.download_file(file_list[i], os.path.join(path_others, file_list[i].split('/')[-1]))
    # file_list.pop(i)
    if i % 20000 == 0:
        print('Now has exported %d'%(i))
        print('Now has used %.2f minutes'%((time.time() - start_time)/60))
        # df = pd.DataFrame(file_list)
        # df.to_csv(os.path.join(os.path.join(path, meta_path), meta_name+ str(i)))



# my own bucket
# access_key = 'AKIAT53VSMFE6JVKCE6I'
# secret_key = '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'
# bucket_name = 'lugq-2019-test'

### This is to upload the MML server code to S3 and downlad the zip file into local server
import boto3

# this is alice dev env
access_key = 'AKIARLSQS4QER3F2RWIB'
secret_key = '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'
bucket_name = '30899-aliceportal-dev'


session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
s3 = boto3.resource('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)

folder_name = 'test'
my_bucket = s3.Bucket(bucket_name)


for file in my_bucket.objects.filter(Prefix=folder_name + '/'):
    print(file.key)

path = 'C:/Users/guangqiiang.lu/Documents/lugq/github'
file_name = 'new_hr_code.zip'

client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
client.upload_file(os.path.join(path, file_name), bucket_name, folder_name+'/'+file_name)



import boto3
import os

# this is alice dev env
# access_key = 'AKIARLSQS4QER3F2RWIB'
# secret_key = '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'
# bucket_name = '30899-aliceportal-dev'


# this is my own s3 bucket, I have tested that I could login with these keys
access_key = 'AKIAIEXQMBIYS3MKNM2Q'
secret_key = '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'
bucket_name = 'lugq-2019-test'


session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
s3 = boto3.resource('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)

folder_name = 'test'
my_bucket = s3.Bucket(bucket_name)

for file in my_bucket.objects.filter(Prefix=folder_name + '/'):
    if 'new_hr_code.zip' in file.key:
        print("Get file: ", file.key)
        my_bucket.download_file(file.key, 'new_hr_code.zip')

os.system('unzip new_hr_code.zip')


# import numpy as np
#
# a = np.array(['a', 'a', 'a', 'c', 'c', 'd', 'd'])
# b_org = np.array(['', 's', 'b', 'a', 'b', 'b', 'a'])
# convert_b_to_int = {'s':0, 'b':1, 'a':2, '':3}
# int_to_str = {v: k for k, v in convert_b_to_int.items()}
# b = np.array([convert_b_to_int[x] for x in b_org])
#
# a_set = set(a)
#
# re = ''
# for x in a_set:
#     # get each unique value
#     t = a == x
#     a_list = a[t]
#     b_list = b[t]
#     b_min = np.argmin(b_list)
#     b_value = int_to_str[b_list[b_min]]
#     re += (b_value + a_list[b_min] + '|')
#

# import os
# import pandas as pd
# import numpy as np
#
# sp_cols = ['supply_bulk.specialization1','supply_bulk.specialization2','supply_bulk.specialization3','supply_bulk.specialization4','supply_bulk.specialization5','supply_bulk.specialization6','supply_bulk.specialization7','supply_bulk.specialization8']
# ac_cols = ['supply_bulk.assessmenttype1','supply_bulk.assessmenttype2','supply_bulk.assessmenttype3','supply_bulk.assessmenttype4','supply_bulk.assessmenttype5','supply_bulk.assessmenttype6','supply_bulk.assessmenttype7','supply_bulk.assessmenttype8']
#
# path = 'C:/Users/guangqiiang.lu/Documents/lugq/workings/202001'
#
# df = pd.read_csv(os.path.join(path, [x for x in os.listdir(path) if x.endswith('csv')][0]), sep='\x1F',quoting=3)
#
# df1 = df[sp_cols].fillna('').values
# df2 = df[ac_cols].fillna('').values
#
# assert len(df1) == len(df2)
#
# convert_b_to_int = {'s':0, 'b':1, 'a':2, '':3}
# int_to_str = {v: k for k, v in convert_b_to_int.items()}
#
#
# def get_first_car(value_list):
#     return [x[0].lower() if x != '' else x for x in value_list]
#
# # loop for each person
# res = []
# for i in range(len(df1)):
#     sp_cols = df1[i, :]
#     ac_cols = df2[i, :]
#     ac_cols = get_first_car(ac_cols)
#     ac_cols = np.array([convert_b_to_int[x] for x in ac_cols])
#
#     sp_set = set(sp_cols)
#     # loop for each unique value
#     t = ''
#     for v in sp_set:
#         if v == '':
#             # if we just get nan value in first dataframe just append null string
#             t += ''
#         else:
#             sati = sp_cols == v
#             sp_cols_new = sp_cols[sati]
#             ac_cols_new = ac_cols[sati]
#             ac_min = np.argmin(ac_cols_new)
#             ac_value = int_to_str[ac_cols_new[ac_min]]
#             if ac_value == '':
#                 continue
#             t += ac_value + sp_cols_new[ac_min] + '|'
#     res.append(t)
