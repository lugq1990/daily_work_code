# -*- coding:utf-8 -*-
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('echo', help='just give number', type=int)
# here is just to add the optional args
parser.add_argument('--verb', help='just add the verb', type=int)
args = parser.parse_args()

if args.verb is not None:
    print('Now is {}'.format(args.verb**2))





"""This is just to test that for the code transfer have run successfully"""
import boto3
sftp_path = '/sftp/cio.alice'

access_key = 'AKIAJGK5GH7CU46OSS5Q'
secret_key = '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'
bucket_name = 'aliceportal-30899-prod'
s3_folder_name = 'Delta'
session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
client = session.client('s3')
s3 = session.resource('s3')
my_bucket = s3.Bucket(bucket_name)

import paramiko
host = '10.5.105.51'
port = 22
username = 'ngap.app.alice'
password = 'QWer@#2019'
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

ssh.connect(host, port=port, username=username, password=password, look_for_keys=False)

sftp = ssh.open_sftp()
already_download_list = sftp.listdir(sftp_path)

file_list = []
for f in my_bucket.objects.filter(Prefix=s3_folder_name+'/'):
    if f.key.endswith('.txt'):
        file_list.append(f.key)
    if len(file_list) % 50000 == 0:
        print('Here get {} files'.format(len(file_list)))
print('Total files {}'.format(len(file_list)))

file_list = [x.split('/')[1] for x in file_list]
# here is to check both the source will get the same files list
diff_file_list = set(file_list) - set(already_download_list)
print('There are {} files not downloaded'.format(len(diff_file_list)))



# cause so many files couldn't be put to HDFS with put command, here just create 10 folders, move files
path = '/sftp/cio.alice/missing_orc'
import os
folder_list = ['data_0',  'data_1',  'data_2',  'data_3',  'data_4',  'data_5',  'data_6',  'data_7',  'data_8',  'data_9',
               'data_A',  'data_B',  'data_C',  'data_D',  'data_E',  'data_F',  'data_G',  'data_H',
               'data_I',  'data_J',  'data_K',  'data_L',  'data_M',  'data_N',  'data_O',  'data_P',
               'data_Q',  'data_R',  'data_S',  'data_T',  'data_U',  'data_V',  'data_W',  'data_X',
               'data_Y',  'data_Z', 'others']

file_list = os.listdir('/sftp/cio.alice')
file_list = [x for x in file_list if x.endswith('.txt')]

file_ends = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
file_ends_list = [x for x in file_ends]

from shutil import copyfile
i = 0
for fi in file_list:
    if len(fi) <= 5:
        continue
    if fi[-5].upper() not in file_ends_list:
        copyfile(os.path.join('/sftp/cio.alice', fi), os.path.join(os.path.join(path, folder_list[-1]), fi))
    else:
        des_folder = 'data_' + fi[-5].upper()
        copyfile(os.path.join('/sftp/cio.alice', fi), os.path.join(os.path.join(path, des_folder), fi))
    i += 1
    if i % 5000 == 0:
        print('already moved %d files'%(i))

for fo in folder_list:
    print('Now is folder:', fo)
    os.system('hdfs dfs -put -f %s/*.txt /data/insight/cio/alice/contracts_files/20190530/ '%(os.path.join(path, fo)))
    os.system('hdfs dfs -put -f %s/*.txt /data/insight/cio/alice/contracts_files/whole_files/ '%(os.path.join(path, fo)))



whole_file_list = []
for fo in folder_list:
    whole_file_list.extend(os.listdir(os.path.join(path, fo)))