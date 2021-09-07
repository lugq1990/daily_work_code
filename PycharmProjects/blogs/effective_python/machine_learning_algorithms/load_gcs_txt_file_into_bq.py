# -*- encoding: utf-8 -*-
'''
Try to load a txt file with some headers, but try to use cloud fucntions to load them into BQ.


Load txt data from gcs and processing them into a DataFrame then load it directly into BQ.
@time: 2021/05/31 15:54:38
@author: Guangqiang.lu
'''
import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from google.cloud import storage
from google.cloud import bigquery

x, y = load_iris(return_X_y=True)

cur_path = os.path.dirname(os.path.abspath(__file__))
data = np.concatenate([x, y[:, np.newaxis]], axis=1)
cols= ['a', 'b', 'c', 'd', 'label']
data_str = ''
for col in data:
    data_str += ','.join([str(x) for x in col]) + "\n"

# write into a txt file
with open(os.path.join(cur_path, 'sample.txt'), 'w') as f:
    f.write(','.join(cols) + "\n")
    f.write(data_str)
    

storage_client = storage.Client()
bq_client = bigquery.Client()

bucket_name = "test_bucket_bq"
file_name = 'sample.txt'
dest_name = "{}.{}".format('iris_data', 'iris')


bucket = storage_client.get_bucket(bucket_name)
blob = bucket.get_blob(file_name)

blob_str = blob.download_as_string().decode('utf-8')
blob_str_split = blob_str.split('\n')
header = blob_str_split[0].replace('\r', '').split(",")
data = [d.replace('\r', '').split(',') for d in blob_str_split[1:]]
# Noted here: if we need to get top-n rows, then just with: data = data[top_n:]
# if you need to get data until last 10 records, then just with: data = data[top_n:-last_n]
# This example is get only 1 -> 149 without the last one that -2 means that we don't get last one.
top_n = 1
last_n = -2
data = data[top_n:last_n]

remake_df = pd.DataFrame(data, columns=header)

# I find that to use `to_gbq` is easier if we already have our table created.
# Keep in mind that we need to install `pandas-gbq`
remake_df.to_gbq(dest_name, if_exists='replace')

# job_config= bigquery.LoadJobConfig(schema=[bigquery.SchemaField(x, 'string') for x in cols], write_disposition='write_truncate')

# # This needed pyarrow to be installed. Do keep in mind to install pyarrow==4.0.0
# # https://stackoverflow.com/questions/66892409/cant-install-pyarrow-on-ubuntu
# job_res = bq_client.load_table_from_dataframe(remake_df, dest_name, job_config=job_config)
# job_res.result()

table = bq_client.get_table(dest_name)
print("Get {} records in table: {}".format(table.num_rows, dest_name))


