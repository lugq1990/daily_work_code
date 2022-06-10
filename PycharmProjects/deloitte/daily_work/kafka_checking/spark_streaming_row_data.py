"""Read kafka message with value string hashed, partition by ds and batch_id"""
from pyspark.sql import SparkSession
from pyspark.sql import functions
from datetime import datetime
from hashlib import md5
from pyspark.sql.types import StringType


spark = SparkSession.builder.getOrCreate()

spark.sparkContext.setLogLevel('warn')

# noted: kafka.group.id is noly useful for spark 3.0v
df = spark\
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "topic_spark") \
    .option("startingOffsets", "earliest") \
    .load()


def hash_value(data):
    if isinstance(data, str):
        data = data.encode('utf-8')
    return md5(data).hexdigest()


hash_value = functions.udf(hash_value, StringType())


def write_to_file(df, batch_id):
    """
    write_to_file write Dataframe into local disk.

    :param df: Dataframe
    :type df: _type_
    :param batch_id: which batch_id it belongs to.
    :type batch_id: _type_
    """
    print('get batch_id', batch_id)
    now = datetime.now()

    date_str = now.strftime('%Y%m%d')
    df = df.withColumn("batch_id", functions.lit(batch_id)
                       ).withColumn("ds", functions.lit(date_str))

    # add with hash value
    # df = df.withColumn("hash_value", hash_value(functions.col("value")))

    # add current time string
    # time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    # df = df.withColumn("current_time", functions.lit(time_str))
    
    # only get what we want
    # df = df.select(["ds", "batch_id", "hash_value"])    

    df.write.mode('append').partitionBy("ds", "batch_id").json("batch_out_row")


def spark_running():
    # with each topic should with each checkpiont!
    query = df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)") \
        .writeStream \
        .option("checkpointLocation", "topic_spark_checkpoint_row")\
        .trigger(processingTime="10 seconds") \
        .foreachBatch(write_to_file)\
        .start()

    query.awaitTermination()
    # /Users/guangqianglu/Downloads/big_data/spark-3.0.3-bin-hadoop2.7/bin/spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.0.3 /Users/guangqianglu/Documents/work/code_work/daily_work_code/PycharmProjects/deloitte/daily_work/kafka_checking/spark_streaming_row_data.py


if __name__ == "__main__":
    spark_running()
