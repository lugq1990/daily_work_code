# -*- coding:utf-8 -*-
from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
import time
from pyspark.streaming.kafka import KafkaUtils

""" If we use the basic spark streaming to do jobs, we may want to make check_point for it,
    this is the exmample for how to use checkpoint"""

checkpointPath = '/home/luguangqiang/checkpoint'

def createContext(timeDuration=10):
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext.getOrCreate()
    ssc = StreamingContext(sc, timeDuration)
    # set checkpoint directory
    ssc.checkpoint(checkpointPath)
    return ssc

ssc = StreamingContext.getOrCreate(checkpointPath, createContext)

brokers = 'de-bdt-kafka01:6667,de-bdt-kafka02:6667,de-bdt-kafka02:6667'
topics = 'test2'

kvs = KafkaUtils.createDirectStream(ssc, [topics], {'metadata.broker.list': brokers})

lines = kvs.map(lambda x: x[1])
counts = lines.flatMap(lambda line: line.split(' ')) \
    .map(lambda word: (word, 1)) \
    .groupByKey(lambda a, b: a + b)

print('Now is ', time.time())
print('Result :')
print('*' * 50, type(counts))
counts.pprint()

ssc.start()
ssc.awaitTermination()
