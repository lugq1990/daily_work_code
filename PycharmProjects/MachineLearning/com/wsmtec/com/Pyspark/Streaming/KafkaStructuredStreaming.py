# -*- coding:utf-8 -*-
"""
    This is just an example for how to use Structured Streaming to do the word count
    reading data from kafka
"""

import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split
from pyspark.sql.functions import window

if __name__ == '__main__':
    # if len(sys.argv) != 4:
    #     print('Not good params! ')
    #     sys.exit(-1)

    bootstrapServer = 'de-bdt-kafka01:6667,de-bdt-kafka02:6667,de-bdt-kafka02:6667'
    pull_topics = 'test2'
    push_topics = 'replicated'

    spark = SparkSession.builder.appName('Kafka_examles').getOrCreate()
    spark.conf.set("spark.sql.streaming.checkpointLocation", "/home/luguangqiang/pyspark")

    # using spark to read the stream as DataSet from kafka
    lines = spark.readStream.format('kafka') \
        .option('kafka.bootstrap.servers', bootstrapServer)\
        .option('subscribe', pull_topics)\
        .load() \
        .selectExpr(['cast(value as string)','timestamp'])

    # split the lines to words
    words = lines.select(explode(split(lines.value, ' ')).alias('word'))
    print('*'*100)
    words.printSchema()

    # generate the running word count
    wordsCount = words.withWatermark("timestamp", "10 minutes") \
        .groupBy(
            window(words.timestamp, "10 minutes", "5 minutes"),
            words.word)\
        .count()

    # sinks the result to Kafka, after calling the start(), DataSet will write to the given topic
    ds = wordsCount.writeStream.format('kafka')\
        .option('kafka.bootstrap.servers', bootstrapServer)\
        .option('subscribe', push_topics)\
        .start()

    # print('This is the Result:')
    # # ds = wordsCount.writeStream.outputMode('complete').format('console').start()
    # ds = wordsCount.writeStream\
    #     .outputMode('complete')\
    #     .format('csv')\
    #     .option('path', '/home/luguangqiang/pyspark')\
    #     .start()


    # if there is already an active streaming still running, await for a second
    ds.awaitTermination()




