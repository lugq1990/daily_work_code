# -*- coding:utf-8 -*-
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, window, get_json_object
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StringType, ArrayType
from pyspark.sql.functions import col


""" this is an example for how to use the continuous processing engine """

# this function is used to make sure only one spark in jvm
def getSparkInstance():
    if ('spark' not in globals()):
        globals()['spark'] = SparkSession\
            .builder\
            .enableHiveSupport()\
            .config("spark.sql.streaming.checkpointLocation", "/home/luguangqiang/pyspark/checkpoint/ck")\
            .appName('new_kafka').getOrCreate()
    return globals()['spark']

spark = getSparkInstance()

# spark.conf.set("spark.sql.streaming.checkpointLocation", "/home/luguangqiang/pyspark/checkpoint/ck")

bootstrapServer = 'de-bdt-kafka01:6667,de-bdt-kafka02:6667,de-bdt-kafka02:6667'
pull_topics = 'test2'
push_topics = 'replicated'
groupid = 'lu'

# before to get the json object data, first to construct the json structure
schema = StructType().add('name', StringType()).add('age', StringType())\
    .add('city', StringType()).add('sex', ArrayType(StringType()))

lines = spark.readStream.format('kafka')\
    .option('kafka.bootstrap.servers', bootstrapServer)\
    .option('subscribe', pull_topics) \
    .option("startingOffsets", "earliest")\
    .load()

words = lines\
    .withWatermark('timestamp', '10 seconds')\
    .select(window(col('timestamp'),  "10 seconds", "10 seconds"), col('value'))\
    .select(from_json(col('value').cast('string'), schema).alias('data'))\
    .select("data.*")
    # .select(
    #     window(lines.timestamp, "10 seconds", "10 seconds"),
    #     lines.value)

# This time, I use the extract the json struture data
# words = df.select(from_json(col('value').cast('string'), schema).alias('data')).select("data.*")

# words = lines.select(explode(split(lines.value, " ")).alias('value'))

# register the words dataframe to a table
words.createOrReplaceTempView('t')

res = spark.sql("""select name, age, city, collect_list(sss) as sex_num from
    (select name, age, city, explode(sex) as sss from t)t2 group by name, age, city""")

# res = words.select(col('name'), col('age'), col('city'), explode(col('sex')).alias('ss'))

# write the result to kafka
query = res\
    .selectExpr("to_json(struct(*)) AS value")\
    .writeStream\
    .format('kafka')\
    .outputMode('append')\
    .option('kafka.bootstrap.servers', bootstrapServer)\
    .option('subscribe', push_topics)\
    .option('topic', push_topics)\
    .start()

query.awaitTermination()




