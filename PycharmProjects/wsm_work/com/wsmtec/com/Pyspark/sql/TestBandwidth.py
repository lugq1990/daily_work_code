# -*- coding:utf-8 -*-
from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
import time

a = np.arange(100000).reshape(-1,1)
b = pd.DataFrame(a, columns=['id'])

spark = SparkSession.builder.getOrCreate()

df = spark.createDataFrame(b)

df.createTempView('t')
spark.sql('show databases').show()
s_t = time.time()
spark.sql('insert into table lugq.test_bandwidth select id from t')
e_t = time.time()
print('Finished! total use {0:.4f} seconds'.format(e_t - s_t))
