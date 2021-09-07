# -*- coding:utf-8 -*-
"""
This is a demo to demonstrate the use case of spark

@author: Guangqiang.lu
"""
from pyspark.sql import SparkSession
from pyspark import SparkConf

conf = SparkConf().setMaster("local[*]").setAppName("test")
spark = SparkSession.builder.config(conf=conf).getOrCreate()
sc = spark.SparkContext

path = "C:/Users/guangqiiang.lu/Documents/lugq/github/spark-master/README.md"

lines = sc.textFile(path)
words = lines.flatMap(lambda x: x.split(" "))
counts = words.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)
counts.collect()

