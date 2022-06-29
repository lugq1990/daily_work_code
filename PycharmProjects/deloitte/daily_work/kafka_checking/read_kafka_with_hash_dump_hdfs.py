"""Read kafka message with value string hashed, partition by ds and batch_id"""

from pyspark.sql import SparkSession
from pyspark.sql import functions
from datetime import datetime
from hashlib import md5
from pyspark.sql.types import StringType

import argparse


broker = "IT-Kafka-Node01:9092"
hdfs_root_path = "hdfs://10.11.16.36:8020/user/root/checkpoint_monitor"
hash_value_output_path = hdfs_root_path + "/hash_value_kafka"

global output_path
global spark


def init_spark(app_name=None):
    """change this for changing spark application name."""
    if not app_name:
        app_name = "kafka_monitor"
    spark = SparkSession.builder.appName(app_name)\
        .config("spark.executor.memoryOverhead", 2048)\
            .config("spark.driver.memoryOverhead", 2048)\
                .config("spark.streaming.kafka.maxRatePerPartition", 1000)\
                    .config("spark.streaming.backpressure.enabled", "true").getOrCreate()
    # spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel('warn')
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    return spark


def hash_value(data):
    if isinstance(data, str):
        data = data.encode('utf-8')
    return md5(data.encode('utf-8')).hexdigest()


# def convert_timestamp_to_str(timestamp, date_format='%Y-%m-%d'):
#     "convert timestamp to date_format str."
#     return datetime.strftime(timestamp, date_format)
    
def convert_timestamp_to_str(timestamp):
    "Current version of spark not support datetime convert, so just try to use string."
    time_str = str(timestamp)
    if " " in time_str:
        date_str = time_str.split(" ")[0]
    else:
        date_str = ""
    return date_str


hash_value = functions.udf(hash_value, StringType())
convert_timestamp_to_str = functions.udf(convert_timestamp_to_str, StringType())


def convert_df(df, batch_id=None):
    """Main dataframe transformation happens here:
        - add batch_id
        - hash value string
        - add with timestamp string
        - select columns are needed.
    """
    now = datetime.now()
    date_str = now.strftime('%Y%m%d')
    
    df = df.withColumn("ds", functions.lit(date_str))
    
    print('get batch_id', batch_id)
    if batch_id is not None:
        # whether or not to add batch_id  
        df = df.withColumn("batch_id", functions.lit(batch_id))

    # add with hash value
    df = df.withColumn("hash_value", hash_value(functions.col("value")))

    # add current time string
    # time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    # df = df.withColumn("current_time", functions.lit(time_str))
    
    # add with timestamp string with udf
    df = df.withColumn("timestamp_str", convert_timestamp_to_str(functions.col('timestamp')))
    
    # only get what we want
    selected_cols = ["ds", "batch_id", "timestamp_str", "hash_value", "value"]
    if batch_id is None:
        # if there isn't batch_id
        selected_cols.remove('batch_id')
        
    df = df.select(selected_cols)    

    return df


def write_to_file(df, batch_id):
    """
    write_to_file write Dataframe into local disk.

    :param df: Dataframe
    :type df: _type_
    :param batch_id: which batch_id it belongs to.
    :type batch_id: _type_
    """
    df = convert_df(df, batch_id=batch_id)
    
    # output path of converted hash value.
    df.write.mode('overwrite').partitionBy("ds", "batch_id").parquet(output_path)


def spark_running(topic="topic_spark", read_data_mode="latest"):
    # noted: kafka.group.id is noly useful for spark 3.0v
    print("Start to extract message from topic: {}".format(topic))
    df = spark\
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", broker) \
        .option("max.poll.records", 25600) \
        .option("maxOffsetsPerTrigger", "25600")\
        .option("failOnDataLoss", "false")\
        .option("subscribe", topic) \
        .option("startingOffsets", read_data_mode) \
        .load()
        
    # with each topic should with each checkpiont!
    checkpoint_path = hash_value_output_path + "/checkpoint_monitor/" + topic 
    
    query = df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)", "timestamp") \
        .writeStream \
        .option("checkpointLocation", checkpoint_path)\
        .trigger(processingTime="100 seconds") \
        .foreachBatch(lambda df, batch_id: write_to_file(df, batch_id))\
        .start()

    query.awaitTermination()
    # /Users/guangqianglu/Downloads/big_data/spark-3.0.3-bin-hadoop2.7/bin/spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.0.3 /Users/guangqianglu/Documents/work/code_work/daily_work_code/PycharmProjects/deloitte/daily_work/kafka_checking/spark_streaming_kafka_data.py


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
   
    parser.add_argument("--topic", default='topic_spark', type=str, help="Which topic to consume?")   
    args = parser.parse_args()
    
    topic_read = args.topic
    
    output_path = hash_value_output_path + "/{}".format(topic_read)
    # init spark with topic_name
    app_name = "kafka_monitor_{}".format(topic_read)
    init_spark(app_name)

    spark_running(topic=topic_read)
