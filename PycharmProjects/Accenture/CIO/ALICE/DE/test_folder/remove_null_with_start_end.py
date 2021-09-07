# -*- coding:utf-8 -*-
import os
from hdfs.ext.kerberos import KerberosClient
import pandas as pd

client = KerberosClient("http://name-node.cioprd.local:50070")

path = '/anaconda-efs/sharedfiles/projects/alice_30899/data'
file_name = 'DuplicateDocName.xlsx'
hdfs_path = '/data/insight/cio/alice/20190512'

df = pd.read_excel(os.path.join(path, file_name))






