# -*- coding:utf-8 -*-
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import time

if __name__ == '__main__':

    # first to start the spark context
    sc = SparkContext('local[2]')
    ssc = StreamingContext(sc, 5)

    brokers = '10.1.36.61:9092'
    topics = 'test'

    kvs = KafkaUtils.createDirectStream(ssc, [topics], {'metadata.broker.list':brokers})

    lines = kvs.map(lambda x:x[1])
    counts = lines.flatMap(lambda line: line.split(' '))\
        .map(lambda word:(word, 1))\
        .groupByKey(lambda a,b: a+b)

    print('Now is ', time.time())
    print('Result :')
    print('*'*50, type(counts))
    counts.pprint()


    ssc.start()
    ssc.awaitTermination()