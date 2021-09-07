# -*- coding:utf-8 -*-
import numpy as np
from pyspark.sql import SparkSession
import pandas as pd

spark = SparkSession.builder.appName('Test_DF').getOrCreate()

# just random generate a large dataset
x = np.random.random((1000000, 4))
y = np.random.randint(10, size=(1000000, 1))
data = np.concatenate((x, y), axis=1)
# conver the array-like data to be a pandas dataframe
df = pd.DataFrame(data, columns=['a','b','c','d', 'label'])

# make the pandas dataframe to spark dataframe
df_s = spark.createDataFrame(df)

# conver the dataframe's features columns to be vector for mllib
from pyspark.ml.feature import VectorAssembler
df_v = VectorAssembler(inputCols=['a','b','c','d'], outputCol='features').transform(df_s)

# split the dataset to be train and test
(training, test) = df_v.randomSplit([.7, .3])

# build the logistic regression model to fit the dataframe
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(maxIter=100, regParam=.3, elasticNetParam=.8).setFeaturesCol('features').setLabelCol('label')
# start to train the model
model = lr.fit(training)

# get the prediction
pred = model.transform(test)

# evaluate the model
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(metricName='accuracy', labelCol='label', predictionCol='prediction')
acc = evaluator.evaluate(pred)

print('Model Accuracy =%.5f'%acc)
print('Show the prediction')
pred.select('features', 'label','probability', 'prediction').show()


