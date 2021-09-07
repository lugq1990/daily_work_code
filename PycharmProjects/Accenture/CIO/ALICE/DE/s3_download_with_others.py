# -*- coding:utf-8 -*-
import boto3
import os

access_key = 'AKIAJGK5GH7CU46OSS5Q'
secret_key = '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'
bucket_name = 'aliceportal-30899-prod'
whole_s3_folder = 'Delta'
mml_other_s3_folder = '/anaconda-efs/sharedfiles/projects/alice_30899/s3_other_files'
mml_already_downloaded_folder = '/anaconda-efs/sharedfiles/projects/alice_30899/data'

s3 = boto3.Session(aws_secret_access_key=secret_key, aws_access_key_id=access_key).resource('s3')
my_bucket = s3.Bucket(bucket_name)

# This is to get the whole files in S3 bucket
s3_file_list = []
for f in my_bucket.objects.filter(Prefix=whole_s3_folder + '/'):
    if f.key.endswith('.txt'):
        s3_file_list.append(f.key)
    if len(s3_file_list) % 20000 == 0:
        print('Already get %d files'%(len(s3_file_list)))


# This is to get the already downloaded files
# first get all the already download folder
folder_list = [x for x in os.listdir(mml_already_downloaded_folder) if os.path.isdir(os.path.join(mml_already_downloaded_folder, x))]
[folder_list.remove(x) for x in ['doc_id', 'doc_id.txt', 'others']]

already_download_files_list = []
for folder in folder_list:
    print('Now is folder %s'%(folder))
    already_download_files_list.extend(os.listdir(os.path.join(mml_already_downloaded_folder, folder)))


# After get both part files, so for now I have two choice to make the difference between the S3 and local folder
# Change the local folder like S3 or change S3 to alike local folder
# Here just change the local folder to S3, so that for the later download step will the easier
already_download_files_list_converted = [whole_s3_folder+'/'+ x for x in already_download_files_list]

not_download_list = list(set(s3_file_list) - set(already_download_files_list_converted))

# Now that we have get the not download files in the S3, here I will also make the local folder to be like previous
# first step is to create the folder in the destination folder
# this step should just run one time
# for folder in folder_list:
#     os.makedirs(os.path.join(mml_other_s3_folder, folder))


# Now that I have get the not downloaded files list, here I use the boto3 to download the file to local directory
for f in not_download_list:
    exact_folder = 'data_' + f[-5].upper()
    if exact_folder not in folder_list:
        exact_folder = 'data_0'
    my_bucket.download_file(f, os.path.join(os.path.join(mml_other_s3_folder, exact_folder), f.split('/')[1]))
    if not_download_list.index(f) % 5000 == 0:
        print('Already download %d  files'%(not_download_list.index(f)))
print('All files have been downloaded')

# Here I just want to move the whole files that are downloaded to copy to SFTP in production with different folders
# Loop with whole folder
prod_sftp = '/sftp/cio.alice/s3_other_files/'
for f in folder_list:
    os.system("scp %s/%s/*.txt ngap.app.alice@10.5.105.51:%s"%(mml_other_s3_folder, f, prod_sftp))


# this function is used to check whether whole files have been downloaded
# tmp_list = []
# for folder in folder_list:
#     tmp_list.extend(os.listdir(os.path.join(mml_other_s3_folder, folder)))
#     if len(tmp_list) % 10000 == 0:
#         print('Already downloaded %d files'%(len(tmp_list)))
# if len(tmp_list) == len(not_download_list):
#     print('Whole files has been downloaded to the MML serser side')


import paramiko
from paramiko import AuthenticationException

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

host = '10.5.105.51'
port = 22
username = 'ngap.app.alice'
password = 'QWer@#2019'

hdfs_path = '/data/insight/cio/alice/20190508'

try:
    ssh.connect(hostname=host, port=port, username=username, password=password)
    command = "hdfs dfs -put -f %s/*.txt %s"%(prod_sftp, hdfs_path)
    print('Now running command is {}'.format(command))
    ssh.exec_command(command)
except AuthenticationException as e:
    print('Trying to execute with error:%s'%(e))





### this is to count the unnormal data in the HDFS and S3 bucket(here is with the already download files in the MML)
import os
mml_other_s3_folder = '/anaconda-efs/sharedfiles/projects/alice_30899/s3_other_files'
mml_already_downloaded_folder = '/anaconda-efs/sharedfiles/projects/alice_30899/data'
file_list1 = []
file_list2 = []
folder_list = os.listdir(mml_other_s3_folder)
# list whole files in previous and later download files
[file_list1.extend(os.listdir(os.path.join(mml_other_s3_folder, f))) for f in folder_list]
[file_list2.extend(os.listdir(os.path.join(mml_already_downloaded_folder, f))) for f in folder_list]
file_list = file_list1 + file_list2

# here is to get the HDFS folder files
from hdfs.ext.kerberos import KerberosClient
client = KerberosClient("http://name-node.cioprd.local:50070")

hdfs_path = '/data/insight/cio/alice/20190508'
hdfs_list = client.list(hdfs_path)


# here is to get how many files not putted to HDFS
not_put_hdfs_list = list(set(file_list) - set(hdfs_list))


# this is to get the max length of the file
file_length_list = [len(x) for x in file_list]
max_file_length = max(file_length_list)
max_file_name = file_list[file_length_list.index(max_file_length)]


file_hdfs_length_list = [len(x) for x in hdfs_list]
max_file_length = max(file_hdfs_length_list)
max_file_name = hdfs_list[file_hdfs_length_list.index(max_file_length)]


from collections import Counter
top_n_files = Counter(file_hdfs_length_list).most_common()[:5]
normal_list = [x for x in hdfs_list if (len(x) == 16) and '_' not in x]

# get the whole text file without not number or not string
normal_file_list_org = [x[:-4] for x in normal_list]
normal_file_list = normal_file_list_org.copy()
not_normal_list = []

def check_name(x):
    for s in x:
        if not (s.isalpha() or s.isdigit()):
            return True
    return False

for f in normal_file_list_org:
    if check_name(f):
        not_normal_list.append(f)
        normal_file_list.remove(f)



### up step is to get the whole files, but for bellow step is to get the commen files in the HIVE databases
import pandas as pd
file_df = pd.read_csv(os.path.join(mml_other_s3_folder, 'already_download_file.csv'))
hdfs_df = pd.read_csv(os.path.join(mml_other_s3_folder, 'hdfs_list.csv'))
eng_df = pd.read_csv(os.path.join(os.path.join(mml_other_s3_folder, 'hdfs_file.csv'), 'english_file.csv'))

eng_list = eng_df['file_name'].values.tolist()
eng_list = [x+'.txt' for x in eng_list]


# this is to ensure that for hdfs files that aren't combined are just with file name aren't english
not_eng_list = list(set(hdfs_list).intersection(set(eng_list)))

# Overwrite table with spark dataframe
from pyspark.sql import SparkSession
spark = SparkSession.builder.enableHiveSupport.getOrCreate()

path = 'this is spark local path'
file_name = '***.csv'
df = spark.read.option('header', 'true').load(os.path.join(path, file_name))

# save dataframe to Hive table, this operation will overwrite the existing records and append the new records
df.write.mode('overwrite').saveAsTable('alice_uat_staging_30899.fullname')

