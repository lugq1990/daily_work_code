# -*- coding:utf-8 -*-
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split
from pyspark.sql.functions import window

spark = SparkSession.builder.enableHiveSupport().appName('new_kafka').getOrCreate()
spark.conf.set("spark.sql.streaming.checkpointLocation", "/home/luguangqiang/pyspark/checkpoint/ck")

bootstrapServer = 'de-bdt-kafka01:6667,de-bdt-kafka02:6667,de-bdt-kafka02:6667'
pull_topics = 'test2'
push_topics = 'replicated'
groupid = 'lu'

lines = spark.readStream.format('kafka')\
    .option('kafka.bootstrap.servers', bootstrapServer)\
    .option('subscribe', pull_topics)\
    .load()

lines = lines\
    .withWatermark('timestamp', '10 minutes')\
    .select(
        window(lines.timestamp, "10 seconds", "10 seconds"),
        lines.value)

words = lines\
    .select(
   explode(
       split(lines.value, " ")
   ).alias("word")
)

words.createOrReplaceTempView('t')
d_new = spark.createDataFrame([['i1'],['am2'], ['fine2']], ['value'])
# static = spark.sql('select value from p_luguangqiang_db.structured')
wordCounts = spark.sql('select concat(word,value) as value from (select word, count(1) as value from t group by word)t2')
# wordCounts = spark.sql('select word as value from t')

# wordCounts.join(d_new, 'value')



query = wordCounts\
        .writeStream\
        .outputMode('update')\
        .format('kafka')\
        .option('kafka.bootstrap.servers', bootstrapServer)\
        .option('subscribe', push_topics)\
        .option('topic', push_topics)\
        .trigger(continuous="1 second")\
        .start()
#.trigger(processingTime='1 seconds')\

query.awaitTermination()
