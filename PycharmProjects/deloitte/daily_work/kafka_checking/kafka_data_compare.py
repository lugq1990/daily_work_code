"""Have to use a functionality to extract both HDFS and Kafka data with hash values, 
then we could do compare!

But how to do real compare here? 

Maybe should let 2 spark streaming jobs keep run, then this functionality is to load 
data within a period like one day, then we just load them from hashed values.

Things for now is production kafka data is dump without any transformation, so here should
contain a functionality to load dump HDFS data and convert it with hash func then dump it 
to same place like hash_straming function.
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions
from pyspark.sql.types import StringType
from hashlib import md5
from datetime import datetime


spark = SparkSession.builder.getOrCreate()

spark.sparkContext.setLogLevel('warn')


def hash_value(data):
    if isinstance(data, str):
        data = data.encode('utf-8')
    return md5(data).hexdigest()

# spark udf function.
hash_value = functions.udf(hash_value, StringType())


def hash_hdfs_data(row_hdfs_path, dump_hdfs_path="batch_out_hdfs"):
    # load data first
    df = read_data(row_hdfs_path, extracted_col=None)
    
    # add with hash value
    df = df.withColumn("hash_value", hash_value(functions.col("value")))
    
    # only get what we want
    df = df.select(["ds", "batch_id", "hash_value"])    
    
    # dump df to disk
    df.write.mode('overwrite').partitionBy("ds", "batch_id").json(dump_hdfs_path)


def read_data(path, data_type='json', extracted_col='hash_value'):
    df = spark.read.format(data_type).load(path)
    
    # only get columns with extracted_col
    if extracted_col:
        if isinstance(extracted_col, str):
            extracted_col = [extracted_col]        
        df = df.select(extracted_col)
        
    return df


def compare_kafka_and_hdfs(kafka_data_path, hdfs_data_path):
    # load them from disk and do comparation
    kafka_df = read_data(kafka_data_path)
    
    hdfs_df = read_data(hdfs_data_path)
    
    # cache them
    kafka_df.cache()
    hdfs_df.cache()
    
    # get diff
    kafka_diff = kafka_df.exceptAll(hdfs_df)
    hdfs_diff = hdfs_df.exceptAll(kafka_df)
    
    # get diff num
    kafka_diff_num = kafka_diff.count()
    hdfs_diff_num = hdfs_diff.count()
    
    return (kafka_diff_num, hdfs_diff_num)


if __name__ == "__main__":
    row_hdfs_path = "/Users/guangqianglu/Documents/work/code_work/daily_work_code/PycharmProjects/batch_out_row"
    kafka_data_path = "/Users/guangqianglu/Documents/work/code_work/daily_work_code/PycharmProjects/batch_out"
    hdfs_data_path = "/Users/guangqianglu/Documents/work/code_work/daily_work_code/PycharmProjects/batch_out_hdfs"
    # first let's convert hdfs data
    
    hash_hdfs_data(row_hdfs_path)
    
    kafka_diff_num, hdfs_diff_num = compare_kafka_and_hdfs(kafka_data_path, hdfs_data_path)
    
    print("kakfa_diff_num:", kafka_diff_num)
    print("hdfs_diff_num:", hdfs_diff_num)
    