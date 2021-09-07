# -*- coding:utf-8 -*-
"""Just to query the document name in HDFS"""
from hdfs.ext.kerberos import KerberosClient
import os
import boto3

hdfs_parent_path = '/data/insight/cio/alice/contracts_files/'
folder_part_one = 'whole_files'
folder_part_two = 'whole_files2'

client = KerberosClient("http://name-node.cioprd.local:50070")

file_list = []
folder_list = [os.path.join(hdfs_parent_path, folder_part_one), os.path.join(hdfs_parent_path, folder_part_two)]

for folder in folder_list:
    print('Now is: ', folder)
    file_list.extend(client.list(folder))

access_key = 'AKIAJGK5GH7CU46OSS5Q'
secret_key = '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'
bucket_name = 'aliceportal-30899-prod'
folder_name = 'TextFiles'

session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
s3_client = session.client('s3')
s3 = session.resource('s3')
my_bucket = s3.Bucket(bucket_name)

### don't need this.
file_list_s3 = []
for file in my_bucket.objects.filter(Prefix=folder_name + '/'):
    if file.key.endswith('.txt'):
        file_list_s3.append(file.key)
    if len(file_list_s3) % 100000 == 0:
        print('Already Get %d files.' % len(file_list_s3))

# Just get the file names list
file_list_s3 = [x.split('/')[-1] for x in file_list_s3]

# here to get the files name that contains in the s3 bucket but not in HDFS
diff_list = list(set(file_list_s3) - set(file_list))

# there are so many files in the S3 but not in the list
# here just write the logic to get the file contents and upload the files to HDFS
diff_list = [folder_name + '/' + x for x in diff_list]

# download the files to a temperate folder
# In fact, here I don't need to write the file to local temperate folder, just use
# client to write files in HDFS
# upload the files to WholeFiles2
for f in diff_list:
    file_contents = s3_client.get_object(Bucket=bucket_name, Key=f)['Body'].read()  # in memory
    # write data in HDFS folder
    file_name = f.split('/')[-1]
    client.write(hdfs_path=os.path.join(folder_list[-1], file_name), data=file_contents, overwrite=True)
    if diff_list.index(f) % 5000 == 0:
        print("Already process %d files " % diff_list.index(f))




"""This is to copy the mapping files in s3 bucket and their files to Delta folder in S3"""
import boto3
import os
import numpy as np
import tempfile
import shutil

access_key = 'AKIAJGK5GH7CU46OSS5Q'
secret_key = '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'
bucket_name = 'aliceportal-30899-prod'
folder_name = 'TextFiles'
desc_folder = 'Delta'


session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
s3_client = session.client('s3')
s3 = session.resource('s3')
my_bucket = s3.Bucket(bucket_name)

mapping_file_list = ['MappingFile_20190814_2326.txt', 'MappingFile_20190815_0948.txt', 'MappingFile_20190815_2339.txt']

tmp_path = tempfile.mkdtemp()
mapping_file_list = ['/'.join([folder_name, x]) for x in mapping_file_list]

# download the mapping file to local
for f in mapping_file_list:
    my_bucket.download_file(f, os.path.join(tmp_path, f.split('/')[-1]))


file_list = []
for file in mapping_file_list:
    with open(os.path.join(tmp_path, file.split('/')[-1]), 'r') as f:
        data = [x.split('\t')[0] for x in f.readlines()]
        file_list.extend(data)

file_list = [x + '.txt' for x in file_list]

already_download = []
for f in my_bucket.objects.filter(Prefix="Delta" + '/'):
    if f.key.endswith('.txt'):
        already_download.append(f.key)
    if len(already_download) % 2000 == 0:
        print('Already get %d files. ' % len(already_download))

already_download = [x.split('/')[-1] for x in already_download]

file_list = list(set(file_list) - set(already_download))

file_list = ['/'.join([folder_name, x]) for x in file_list]
file_list.extend(mapping_file_list)

desc_list = [x.replace(folder_name, desc_folder) for x in file_list]

for i in range(len(file_list)):
    src_dirc = {'Bucket': bucket_name, 'Key': file_list[i]}
    s3.meta.client.copy(src_dirc, bucket_name, desc_list[i])
    if i % 500 == 0:
        print('Already copy % d files.' % i)

print('remove the temp folder!')
shutil.rmtree(tmp_path)







already_download = []
for f in my_bucket.objects.filter(Prefix="Delta" + '/'):
    if f.key.endswith('.txt'):
        already_download.append(f.key)
    if len(already_download) % 2000 == 0:
        print('Already get %d files. ' % len(already_download))


print('There are %d files in Delta folder.' % len(already_download))
print('original has %d files ' % (len(file_list) + len(mapping_file_list)))




