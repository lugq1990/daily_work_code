# -*- coding:utf-8 -*-
"""This is used to compare two different files in the HDFS, here is just assume that
 the file is just one column"""
from hdfs.ext.kerberos import KerberosClient
import os
import tempfile
import pandas as pd
import shutil

hdfs_parent_path = "/tmp"
file_name_1 = "test1.txt"    # file that is with more contents
file_name_2 = "test2.txt"
local_store_path = "/mrsprd_data/Users/ngap.app.alice/shell_test/hdfs_file_compare_re"   # which folder to put the result file

# init client
client = KerberosClient("http://name-node.cioprd.local:50070")

# make one temperate folder
tmp_folder = tempfile.mkdtemp()

# download these two files to temperate folder
file_list = []
file_list.append(os.path.join(hdfs_parent_path, file_name_1))
file_list.append(os.path.join(hdfs_parent_path, file_name_2))

# download file to the temperate folder
for f in file_list:
    client.download(f, os.path.join(tmp_folder, f.split('/')[-1]))

# after download step finished, open the two files and compare with each other
with open(os.path.join(tmp_folder, file_name_1), 'r') as f:
    file_content_1 = list(f.readlines())

with open(os.path.join(tmp_folder, file_name_2), 'r') as f2:
    file_content_2 = list(f2.readlines())

file_content_1 = [x.replace('\n', '') for x in file_content_1]
file_content_2 = [x.replace('\n', '') for x in file_content_2]

# get different contents, ensure that shouldn't with duplicate records
diff_list = list(set(file_content_1) - set(file_content_2))

df = pd.DataFrame(diff_list, columns=['diff_records'])

# save different records file to local folder
df.to_csv(os.path.join(local_store_path, 'diff_records.csv'), index=False)

# after whole step finished, just remove the temperate folder
shutil.rmtree(tmp_folder)

print("Whole step finished!")
