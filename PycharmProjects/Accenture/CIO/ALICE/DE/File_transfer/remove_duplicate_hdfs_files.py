# -*- coding:utf-8 -*-
import os
from hdfs.ext.kerberos import KerberosClient
import pandas as pd

client = KerberosClient("http://name-node.cioprd.local:50070")

path = '/anaconda-efs/sharedfiles/projects/alice_30899/data'
file_name = 'DuplicateDocName.csv'
hdfs_path = '/data/insight/cio/alice/20190512'

# here file is just one column with 'file_name'
df = pd.read_csv(os.path.join(path, file_name))
file_list = df['file_name'].values.tolist()

# just to get file name without extensions
file_list = [x.strip().split('.')[0] for x in file_list]

whole_file_list = client.list(hdfs_path)
org_file_number = len(whole_file_list)

print('Original files number: {}'.format(len(whole_file_list)))
### 978429

# loop for whole files and try to remove the satisfied files
remove_result = []
for f in file_list:
    try:
        remove_result.append(client.delete(os.path.join(hdfs_path, f + '.txt')))
    except Exception as e:
        pass
    if file_list.index(f) % 5000 == 0:
        print('Already removed {} files'.format(file_list.index(f)))

import numpy as np
print('Get satisfied files {} have been removed from HDFS'.format(np.sum(np.array(remove_result))))
## 26056



df_hdfs = pd.DataFrame([x.split('.')[0] for x in whole_file_list])
df_hdfs.to_csv(os.path.join(path, 'hdfs_whole_files.csv'), header=False, index=False)

df_hive = pd.DataFrame(np.array([x.split('.')[0] for x in df['file_name'].values.tolist()]))
df_hive.to_csv(os.path.join(path, 'hive_whole_files.csv'), header=False, index=False)

df_hdfs.columns = ['file_name']
df_hive.columns = ['file_name']

out_df = pd.merge(df_hive, df_hdfs, on='file_name', how='inner')

file_list = [x for x in os.listdir(path) if x.endswith('.csv')]