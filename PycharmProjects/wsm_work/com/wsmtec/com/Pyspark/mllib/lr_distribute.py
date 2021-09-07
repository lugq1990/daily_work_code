# -*- coding:utf-8 -*-
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

path = '/user/luguangqiang/mllib'
df = spark.read.format('libsvm').load(path + '/sample_libsvm_data.txt')
# split the data to train and test
(training, test)  = df.randomSplit([.7, .3])

lr = LogisticRegression(maxIter=100, regParam=.3, elasticNetParam=.8)

# start to train the model
model = lr.fit(training)

pred = model.transform(test)

print('show the result: ')
pred.show()

