from google.cloud import storage
import os

sa_file = "ServiceKey_DSDBD_storage.json"

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = sa_file

client = storage.Client()

bucket_name = "npd-65343-datalake-bd-11811-dsd-npd-bd-ca-dsdgcs-raw"

bucket = client.get_bucket(bucket_name)

file_list = list(bucket.list_blobs())

print("File list:", [x.name for x in file_list])


# os.makedirs("tmp", exist_ok=True)

import pandas as pd


error_file_list = []

gcs_file_list = ["gs://" + bucket_name + "/" + f.name for f in file_list]

for f in gcs_file_list:
    try:
        df = pd.read_csv(f, nrows=10, delimiter='|')
        if df.shape[1] < 2:
            print("Get error file:{}".format(f))
            error_file_list.append(f)
    except Exception as e:
        error_file_list.append(f)
        print("When to read file:{} get error:{}".format(f, e))

df = pd.read_csv("gs://npd-65343-datalake-bd-11811-dsd-npd-bd-ca-dsdgcs-raw/Resource_india.csv", nrows=10, delimiter='|')




for blob in file_list:
    file_name = blob.name
    try:
        blob.download_to_filename(os.path.join('tmp', file_name))
        print("Already download: {}".format(file_name))
        try:
        # print("To load file:{}".format(f))
            df = pd.read_csv(os.path.join('tmp', file_name), delimiter='|', nrows=10)
            if df.shape[1] < 2:
                error_file_list.append(file_name)
                print("Not right file: {}".format(file_name))
        except Exception as e:
            error_file_list.append(file_name)
            print("When to load file:{} get error: {}".format(file_name, e))
        os.remove(os.path.join('tmp', file_name))
    except Exception as e:
        print("When to download file:{} get error:{}".format(file_name, e))

error_df = pd.DataFrame(error_file_list, columns=['fie_name'])

error_df.to_csv('error_file_name.csv', index=False)


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

df = pd.DataFrame(np.random.randn(10, 2), columns=['a', 'b'])   

df = spark.createDataFrame(df)
df.take(1)