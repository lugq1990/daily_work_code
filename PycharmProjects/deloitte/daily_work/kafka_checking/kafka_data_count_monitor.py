import os
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

broker = "localhost:9092"
topic = "first"

df = spark \
  .readStream \
  .format("kafka") \
  .option("kafka.bootstrap.servers", broker) \
  .option("subscribe", topic) \
  .load()
df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")

df.writeStream.format('console').start()

# /Users/guangqianglu/Downloads/big_data/spark-3.0.3-bin-hadoop2.7/bin/spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.0.3 /Users/guangqianglu/Documents/work/code_work/daily_work_code/PycharmProjects/deloitte/daily_work/kafka_checking/kafka_data_count_monitor.py


