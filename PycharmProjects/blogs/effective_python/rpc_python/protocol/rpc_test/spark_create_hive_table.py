import numpy as np
import pandas as pd

from pyspark.sql import SparkSession


df = pd.DataFrame(np.random.rand(10, 2), columns=['a', 'b'])

spark = SparkSession.builder.getOrCreate()

df = spark.createDataFrame(df)
df.registerTempTable("t1")
spark.sql("create database test")
spark.sql("use test")

spark.sql("create table t2 as select * from t1")
print("Table created")

spark.sql("select * from t2").show()