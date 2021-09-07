# -*- coding:utf-8 -*-
"""this is just to test the hdfs module to append data to one file in HDFS """
from hdfs.ext.kerberos import KerberosClient
import os
import json
import tempfile

client = KerberosClient("http://name-node.cioprd.local:50070")

hdfs_path = '/data/insight/cio/alice/contracts_files/whole_files'
local_path = '/mrsprd_data/Users/ngap.app.alice/shell_test/test_data'
tmp_path = tempfile.mkdtemp()

file_list = client.list(hdfs_path)
import numpy as np
file_np = np.array([x for x in file_list if x.endswith('.txt')])
not_file_np = np.array([x for x in file_list if not x.endswith('.txt')])


file_name_list = []
fake_file_num = 5
for i in range(fake_file_num):
    file_name_list.append('file_' + str(i) + '.txt')

for file in file_name_list:
    with open(os.path.join(tmp_path, file), 'w') as f:
        f.write(json.dumps({file: 'test'+file}))
    f.close()

client.upload(hdfs_path=hdfs_path, local_path=os.path.join(tmp_path, file_name_list[0]))

# After upload one file to the HDFS, then append the whole other files to the original file
for f in file_name_list[1:]:
    with open(os.path.join(tmp_path, f), 'r') as m:
        data = json.load(m)
    m.close()
    client.write(hdfs_path=os.path.join(hdfs_path, file_name_list[0]), data=data, append=True)


