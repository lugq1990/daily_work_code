# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession, SQLContext
from pyspark.ml.classification import LogisticRegression
import tensorflow as tf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

iris = load_iris()
x, y = iris.data, iris.target
sc = SparkContext()
spark = SparkSession.builder.getOrCreate()
# sqlContext = SparkContext(sc)

data = np.concatenate((x, y.reshape(-1,1)), axis=1)
data = pd.DataFrame(data, columns=['a','b','c','d', 'label'])

df = spark.createDataFrame(data)
df_new = VectorAssembler(inputCols=['a','b','c','d'], outputCol='raw_features').transform(df).select('raw_features', 'label')
from pyspark.ml.feature import StandardScaler
df_scaler = StandardScaler(inputCol='raw_features', outputCol='features').fit(df_new).transform(df_new).select('features', 'label')

lr = LogisticRegression(maxIter=100, regParam=.3, elasticNetParam=.8, featuresCol='features',labelCol='label')
model = lr.fit(df_scaler)
pred = model.transform(df_scaler)

evaluator = MulticlassClassificationEvaluator(metricName='accuracy', predictionCol='prediction', labelCol='label')
acc = evaluator.evaluate(pred)

print('The Spark Logistic Regression accuracy = %.7f'%(acc))

