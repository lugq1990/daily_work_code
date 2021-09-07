# -*- coding:utf-8 -*-
"""This is to make some temperate files in the temp folder and put these files to that S3 folder"""
import boto3
import tempfile
import os
import shutil
import random
import string

# this is S3 parameters
access_key = 'AKIAJGK5GH7CU46OSS5Q'
secret_key = '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'
bucket_name = 'aliceportal-30899-prod'
s3_folder_name = 'Delta'
s3 = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key).resource('s3')
my_bucket = s3.Bucket(bucket_name)

# Here I will just random generate a number about 500, and make a temperate folder
# and make random number files in that folder that's needed to be uploaded to S3 folder
file_numbers_mean = 500
file_num_shift = 100
random_file_number = random.randint(file_numbers_mean - random.randint(0, file_num_shift), file_numbers_mean + random.randint(0, file_num_shift))
tmp_path = tempfile.mkdtemp()
file_length = 11
# this is the main random files generator step
file_f = []
for i in range(random_file_number):
    tmp_file_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(file_length))
    rand_text = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(random.randint(1, 100)))
    _, f = tempfile.mkstemp(suffix='.txt', prefix=tmp_file_name, dir=tmp_path, text=True)
    # f = tempfile.NamedTemporaryFile(mode='w', newline=rand_text, suffix='.txt', prefix=tmp_file_name, dir=tmp_path)
    file_f.append(f)
    # write some string to files
    try:
        with open(f, 'w') as tmp:
            tmp.write(rand_text)
        tmp.close()
    except Exception as e:
        pass

# this function should be called just one times
new_create_folder_name = 'WholeTest'
def create_folder_in_s3():
    session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    client = session.client('s3')

    response = client.put_object(
    Bucket=bucket_name,
    Body='',
    Key=new_create_folder_name +'/'
    )
    return response

# this is used to delete files in the S3 BUCKET
def delete_files(folder_name=new_create_folder_name):
    for f in my_bucket.objects.filter(Prefix=folder_name + '/'):
        if f.key.endswith('.txt'):
            s3.Object(bucket_name, f.key).delete()
    return 'Finished deleted'
delete_files()

# after create the folder, here I should put the tempfiles to the folder
for f in file_f:
    my_bucket.upload_file(f, new_create_folder_name+ '/'+f.split('\\')[-1])


file_dict = dict()
for f in my_bucket.objects.filter(Prefix=new_create_folder_name + '/'):
    if f.key.endswith('.txt'):
        file_dict[f.key.split('/')[-1]] = my_bucket.Object(f.key).content_length

import pandas as pd
# create dataframe from dictory
s = pd.Series(file_dict, name='file_size')
s.index.name = 'file_name'
df = s.reset_index()
df.head()


# after whole process finished, I have to delete the temperate folder manually
for f in file_f:
    os.remove(f)
shutil.rmtree(tmp_path)



#### up code is using tempfile to make the file that should updated, for this function,  I use
#### real path to make files, after finished, I will remove the whole files in the folder
def create_file_abs_path(path, file_num):
    tmp_file_list = []
    for i in range(file_num):
        tmp_file_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(file_length))
        rand_text = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(random.randint(1, 100)))
        # _, f = tempfile.mkstemp(suffix='.txt', prefix=tmp_file_name, dir=tmp_path, text=True)
        # # f = tempfile.NamedTemporaryFile(mode='w', newline=rand_text, suffix='.txt', prefix=tmp_file_name, dir=tmp_path)
        # file_f.append(f)
        # write some string to files
        try:
            with open(os.path.join(path, tmp_file_name), 'w') as tmp:
                tmp.write(rand_text)
                tmp_file_list.append(tmp_file_name)
            tmp.close()
        except Exception as e:
            pass
    return tmp_file_list


### this function is used to get the whole files in the S3 bucket folder
import boto3

# this is S3 parameters
access_key = 'AKIAJGK5GH7CU46OSS5Q'
secret_key = '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'
bucket_name = 'aliceportal-30899-prod'
s3_folder_name = 'Delta'
s3 = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key).resource('s3')
my_bucket = s3.Bucket(bucket_name)

file_list = []
for f in my_bucket.objects.filter(Prefix=s3_folder_name+'/'):
    if f.key.endswith('.txt'):
        file_list.append(f.key)
    if len(file_list) % 10000 == 0:
        print('get {} files'.format(len(file_list)))

size_dict = dict()
for f in my_bucket.objects.filter(Prefix=s3_folder_name + '/'):
    if f.key.endswith('.txt'):
        size_dict[f.key.split('/')[-1]] = my_bucket.Object(f.key).content_length

size_list = list(size_dict.values())
import numpy as np
np.sum(np.array(size_list) == 0)

# here is just to test for the production code, copy 10 files from whole folder to Delta folder
des_file_list = [x.replace(s3_folder_name, 'Delta') for x in file_list]
for i in range(len(file_list)):
    src_dict = {'Bucket': bucket_name, 'Key': file_list[i]}
    s3.meta.client.copy(src_dict, bucket_name, des_file_list[i])




"""bellow is to list file in HDFS and update to HDFS folder with HIVE table path"""
# this is used to make the new table files in HDFS,
# and also remove the duplicate files in the wholefile2 folder
from hdfs.ext.kerberos import KerberosClient
import os
import datetime
import tempfile
import pandas as pd
import copy

client = KerberosClient('http://name-node.cioprd.local:50070')

parent_path = '/data/insight/cio/alice/contracts_files'
whole_file_1 = 'whole_files'
whole_file_2 = 'whole_files2'

file_list_whole_1 = client.list(os.path.join(parent_path, whole_file_1))
file_list_whole_2 = client.list(os.path.join(parent_path, whole_file_2))

common_files = list(set(file_list_whole_1) & set(file_list_whole_2))

print('Folder 1:',len(file_list_whole_1))
print('Folder 2:', len(file_list_whole_2))
print('Common folder:',len(common_files))

# first is to remove the common files in the whole_file_2 folder
res_list = []
for f in common_files:
    res_list.append(client.delete(os.path.join(os.path.join(parent_path, whole_file_2), f)))
    if len(res_list) % 5000 == 0:
        print('Already removed %d files '%(len(res_list)))

# after remvoed the duplicated files from HDFS, here is to make the files list in HDFS
doc_hdfs_path = '/data/insight/cio/alice/hivetable/documents_name'

# here is to create one tmp folder, make date string with now
date_str = datetime.datetime.now().strftime('%Y%m%d')
date_str = '20190610'
tmp_folder = tempfile.mkdtemp()


# here just to add one path column
df1 = pd.DataFrame([x[:-4] for x in file_list_whole_1])
df2 = pd.DataFrame([x[:-4] for x in file_list_whole_2])
df1.columns = ['file_name']
df2.columns = ['file_name']
df1['path'] = parent_path + '/' + whole_file_1
df2['path'] = parent_path + '/' + whole_file_2

df_new = pd.concat((df1, df2), axis=0)
df_new['date'] = date_str
df_new = df_new[['file_name', 'date', 'path']]

# remove the file in HDFS
client.delete(hdfs_path=os.path.join(doc_hdfs_path, '{}.csv'.format(date_str)))
client.list(doc_hdfs_path)

# update_file_list = copy.copy(file_list_whole_1)
# update_file_list.extend(list(set(file_list_whole_2) - set(common_files)))
#
# # make one DataFrame with files
# df = pd.DataFrame(update_file_list)
# df.columns = ['file_list']
# df['date'] = date_str

os.system('rm %s/*.csv'%(tmp_folder))
df_new.to_csv(os.path.join(tmp_folder, '{}.csv'.format(date_str)), index=False, header=False)

# after get the files, upload the files to HDFS
client.upload(hdfs_path=doc_hdfs_path, local_path=os.path.join(tmp_folder, '{}.csv'.format(date_str)))




# here is to check with HDFS folder files duplicate
from hdfs.ext.kerberos import KerberosClient
import os
import datetime
import tempfile
import pandas as pd
import copy

client = KerberosClient('http://name-node.cioprd.local:50070')

parent_path = '/data/insight/cio/alice/contracts_files'
date_1 = 'whole_files'
date_2 = 'whole_files2'

file_list1 = set(client.list(os.path.join(parent_path, date_1)))
file_list2 = set(client.list(os.path.join(parent_path, date_2)))

common_files = list(file_list1 & file_list2)






