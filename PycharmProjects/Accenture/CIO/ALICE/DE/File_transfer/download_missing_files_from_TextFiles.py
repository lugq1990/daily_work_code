# -*- coding:utf-8 -*-
import boto3
import os
import paramiko
import pandas as pd

access_key = 'AKIAJGK5GH7CU46OSS5Q'
secret_key = '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'
bucket_name = 'aliceportal-30899-prod'
s3_folder_name = 'TextFiles'

sftp_path = '/sftp/cio.alice'

# init the paramiko
host = '10.5.105.51'
port = 22
username = 'ngap.app.alice'
password = 'QWer@#2019'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

ssh.connect(host, port=port, username=username, password=password, look_for_keys=False)

session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
client = session.client('s3')
s3 = session.resource('s3')
my_bucket = s3.Bucket(bucket_name)

# in case the program fails, here I just list with remote server files first
with ssh.open_sftp() as sftp:
    already_download_list = sftp.listdir()
    already_download_list = [x for x in already_download_list if x.endswith('.txt')]

# sometimes use the pandas to read .csv file with some error, just use engine='python'
file_path_mml = '/mrsprd_data/Users/ngap.app.alice/shell_test/documentname.csv'
df = pd.read_csv('documentname.csv', sep=None, engine='python')
needed_file_list = df.iloc[:, 0].values.tolist()
needed_file_list = [s3_folder_name + '/' + x + '.txt' for x in needed_file_list]


# here first is to get the whole files in s3 TextFiles folder
file_list = []
j = 0
for f in my_bucket.objects.filter(Prefix=s3_folder_name + '/'):
    if f.key.endswith('.txt'):
        file_list.append(f.key)
        j += 1
    if j % 50000 == 0:
        print('get {} files'.format(j))

# get the common file
common_file_list = list(set(file_list).intersection(set(needed_file_list)))

for m in common_file_list:
    file = client.get_object(Bucket=bucket_name, Key=m)['Body'].read()  # write it to memory
    try:
        sftp = ssh.open_sftp()
        with sftp.open(os.path.join(sftp_path, m.split('/')[-1]), 'w') as f_w:
            f_w.write(file)
    except Exception as e:
        continue
    already_download_list.append(m.split('/')[-1])


