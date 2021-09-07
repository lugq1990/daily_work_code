# -*- coding:utf-8 -*-
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth

spark = SparkSession.builder.getOrCreate()

df = spark.createDataFrame([
    (0, [1, 2, 5]),
    (1, [1, 2, 3, 5]),
    (2, [1, 2])
], ["id", "items"])

fp = FPGrowth(itemsCol='items', minConfidence=.5, minSupport=.5)
model = fp.fit(df)

# show the frequent items
print('*'*50)
model.freqItemsets.show()

print('*'*50)
model.associationRules.show()