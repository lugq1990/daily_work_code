# -*- coding:utf-8 -*-
from pyspark.sql import SparkSession


if __name__ == '__main__':
    spark = SparkSession.builder.appName('KafkaToHive').enableHiveSupport().getOrCreate()

    bootstrapServer = '192.168.1.13:6667'
    pull_topics = 'vota_craw_result_topic_20180905'
    table_name = ''

    # using spark to read the stream as DataSet from kafka
    lines = spark\
        .readStream\
        .format('kafka') \
        .option('kafka.bootstrap.servers', bootstrapServer)\
        .option('subscribe', pull_topics)\
        .load() \
        .selectExpr(['cast(value as string)','timestamp'])








    # # split the lines to words
    # words = lines.select(explode(split(lines.value, ' ')).alias('word'))
    # print('*'*100)
    # words.printSchema()
    #
    # # generate the running word count
    # wordsCount = words.withWatermark("timestamp", "10 minutes") \
    #     .groupBy(
    #         window(words.timestamp, "10 minutes", "5 minutes"),
    #         words.word)\
    #     .count()
    #
    # # sinks the result to Kafka, after calling the start(), DataSet will write to the given topic
    # ds = wordsCount.writeStream.format('kafka')\
    #     .option('kafka.bootstrap.servers', bootstrapServer)\
    #     .option('subscribe', push_topics)\
    #     .start()
    #
    #
    #
    # # if there is already an active streaming still running, await for a second
    # ds.awaitTermination()