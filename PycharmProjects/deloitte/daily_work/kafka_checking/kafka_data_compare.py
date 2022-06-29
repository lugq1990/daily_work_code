"""Have to use a functionality to extract both HDFS and Kafka data with hash values, 
then we could do compare!

But how to do real compare here? 

Maybe should let 2 spark streaming jobs keep run, then this functionality is to load 
data within a period like one day, then we just load them from hashed values.

Things for now is production kafka data is dump without any transformation, so here should
contain a functionality to load dump HDFS data and convert it with hash func then dump it 
to same place like hash_straming function.

Things should be noticed: 
    Maybe should add another column for timestamp of message, 
    as there will be a case for data is just overlay for row 
    streaming and monitor streaming. So maybe result is different.
    SO it's recommended to monitor data with previous date as they are constant
    
# table: date, source:kafka/hdfs, value, flag(missing_in_kafka, missing_in_hdfs), table_name?

# todo: extract table name from value, add with date filter.

# monitor: date filter for yesterday.
"""
import os
import json
from pyspark.sql import SparkSession
from pyspark.sql import functions
from pyspark.sql.types import StringType
from hashlib import md5
from datetime import datetime, timedelta


# original streaming dump data path
base_hdfs_path = "hdfs://10.11.16.36:8020/user/hive/stream"
# original streaming with hash func dump path
base_streaming_message_dump_hdfs_path = "hdfs://10.11.16.36:8020/user/root/checkpoint_monitor/hash_value_kafka/original_hash_value"
# kafka already dump hash func values
kafka_message_dump_hdfs_path = "hdfs://10.11.16.36:8020/user/root/checkpoint_monitor/hash_value_kafka"
hive_table_name = "ods_ebs_dev.kafka_monitor_diff"



spark = SparkSession.builder.appName("kafka_data_compare").enableHiveSupport().getOrCreate()
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

spark.sparkContext.setLogLevel('warn')

timestamp_format = '%Y-%m-%d %H:%M:%S.%f'


def hash_value(data):
    if isinstance(data, str):
        data = data.encode('utf-8')
    if data == b"":
        # todo: how to solve value is null?
        return ""
    return md5(data).hexdigest()


# def convert_timestamp_to_str(timestamp, date_format='%Y-%m-%d'):
#     "convert timestamp to date_format str."
#     # first format timestamp to date
#     timestamp = datetime.strptime(timestamp, timestamp_format)
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


def convert_df(df, date_str=None, original_dump_data=False):
    """Main dataframe transformation happens here:
        - add batch_id
        - hash value string
        - add with timestamp string
        - select columns are needed.
    """
    # now = datetime.now()
    # date_str = now.strftime('%Y%m%d')
    
    df = df.withColumn("ds", functions.lit(date_str))   

    # add with hash value
    df = df.withColumn("hash_value", hash_value(functions.col("value")))

    # add current time string
    # time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    # df = df.withColumn("current_time", functions.lit(time_str))
    
    # add with timestamp string with udf
    df = df.withColumn("timestamp_str", convert_timestamp_to_str(functions.col('timestamp')))
    
    # decode value to string
    df = df.withColumn("value", functions.decode(functions.col("value"), "UTF-8"))

    # only get what we want
    selected_cols = ["ds", "batch_id", "timestamp_str", "hash_value", "value"]
    if original_dump_data:
        df = df.withColumnRenamed("batchId", "batch_id")
        
    df = df.select(selected_cols)    

    return df


def _get_date_str(base_on_yesterday=False):
    "used to extract which date folder to read."
    date = datetime.now()
    if base_on_yesterday:
        date = (date - timedelta(days=1))
    date_str = date.strftime('%Y%m%d')
    return date_str


def dump_hash_data_to_hdfs(row_hdfs_path, dump_hdfs_path='batch_out_hdfs', extract_yesterday=False):
    df = read_data(row_hdfs_path, extract_yesterday=extract_yesterday)

    # df.cache()
    
    # we have to give a date string from original path
    date_str = _get_date_str(extract_yesterday)
    print("Get extract date:", date_str)
    df = convert_df(df, date_str=date_str, original_dump_data=True)
    
    # dump df to disk
    print("Start to dump already hash value to path: {}".format(dump_hdfs_path))
    df.coalesce(100).write.mode('overwrite').partitionBy("ds", "batch_id").parquet(dump_hdfs_path)

    # df.unpersist()


def read_data(path, data_type='parquet', extracted_col=None, extract_yesterday=False):
    date_str = datetime.now().strftime('%Y%m%d')
    if extract_yesterday:
        date_str = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
        # add functionality to read yesterday folder!
        if "=" in path:
            path = path.split("=")[0] + "=" + date_str
        else:
            path = os.path.join(path, "ds=" + date_str)
    else:
        # get current date str
        path = path + "/ds={}".format(date_str)
            
    print("Start to read data from path:", path)
    df = spark.read.format(data_type).load(path)

    # we have to add another column named with ds, value with date_str
    df = df.withColumn("ds", functions.lit(date_str))
    
    # remove duplicates records
    df = df.drop_duplicates(subset=['value'])

    # only get columns with extracted_col
    if extracted_col:
        if isinstance(extracted_col, str):
            extracted_col = [extracted_col]        
        df = df.select(extracted_col)
    return df


def write_data_to_hive(df, data_path=None, hive_table_name=None):
    if hive_table_name:
        print("Start to write data to hive table: {}".format(hive_table_name))
        df.write.mode('append').format('hive').saveAsTable(hive_table_name)
    else:
        # dump to file for reference
        df.repartition(1).write.mode('append').format('csv').option('header', "true").save(os.path.join(data_path, "tmp.csv"))


def extract_json(value_str):
    try:
        json_obj = json.loads(value_str)
        if "table" in json_obj:
            return json_obj.get('table')
    except:
        return ""
    

extract_json = functions.udf(extract_json, StringType())


def extract_missing_df(row_df, diff_df, missing_data_source, merge_key='hash_value', ):
    # as we only compare hash value, it's used when we want to get full original diff df
    # noted: Please take care that for missing df, what we need is that to merge the other df
    # example: missing in kafka, then should merge missing df with HDFS df!
    row_df = row_df.drop_duplicates()
    missing_df = row_df.join(diff_df, [merge_key], how='inner')
    
    # add some columns we need.
    
    # which missing data come from
    missing_df = missing_df.withColumn("missing_source", functions.lit(missing_data_source))
    
    # try to extract table name
    missing_df = missing_df.withColumn("table_name", extract_json(functions.col('value')))
    
    #hash_value,timestamp_str,value,ds,batch_id,missing_source,table_name
    selected_cols = ["timestamp_str", "hash_value", "missing_source", "table_name", "value", "ds"]
    missing_df = missing_df.select(selected_cols)
    
    return missing_df



def compare_kafka_and_hdfs(kafka_data_path, 
                           hdfs_data_path,
                           extract_yesterday=False, 
                           save_diff_df=False,
                           save_data_path="/Users/guangqianglu/Downloads/spark_tmp_data",
                           table_name=None):
    # load them from disk and do comparation
    kafka_df = read_data(kafka_data_path, extract_yesterday=extract_yesterday)
    hdfs_df = read_data(hdfs_data_path, extract_yesterday=extract_yesterday)

    # decode key and value for original DF
    # hdfs_df = hdfs_df.withColumn("key", functions.decode(functions.col("key"), "UTF-8")).withColumn("value", functions.decode(functions.col("value"), "UTF-8"))
    
    kafka_hash_df = kafka_df.select("hash_value")
    hdfs_hash_df = hdfs_df.select("hash_value")
    
    # cache them
    kafka_hash_df.cache()
    hdfs_hash_df.cache()
    
    # get diff, this means first df has more than second df
    kafka_diff = kafka_hash_df.exceptAll(hdfs_hash_df)
    hdfs_diff = hdfs_hash_df.exceptAll(kafka_hash_df)
    
    # get diff num
    kafka_diff_num = kafka_diff.count()
    hdfs_diff_num = hdfs_diff.count()
    
    if save_diff_df:
        # save different dataframe 
        if table_name is None and save_data_path is None:
            raise ValueError("When try to save different dataframe, " +
                             "please provide `table_name` and `save_data_path`")
        if kafka_diff_num != 0:
            kafka_missing_df = extract_missing_df(kafka_df, kafka_diff, missing_data_source='hdfs')
            write_data_to_hive(kafka_missing_df, hive_table_name=hive_table_name)
        if hdfs_diff_num != 0:
            hdfs_missing_df = extract_missing_df(hdfs_df, hdfs_diff, missing_data_source='kafka')
            write_data_to_hive(hdfs_missing_df, hive_table_name=hive_table_name)
        
    return (kafka_diff_num, hdfs_diff_num)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--topic", type=str, default='topic_spark',required=True, help="Which topic to read")
    parser.add_argument("--yesterday", type=str, default="no", required=False, help="Whether or not to extract yesterday data?")
    parser.add_argument("--saveDF", type=str, default="no", help="whether or not to save compared different dataframe?")
    
    args = parser.parse_args()
    
    topic = args.topic
    extract_yesterday = str2bool(args.yesterday)
    save_diff_df = str2bool(args.saveDF)
    
    # todo: this should be configurable, try to remove hard code this.
    added_topic_path = "/{}".format(topic)
    row_hdfs_path = base_hdfs_path + added_topic_path
    hash_hdfs_data_path = base_streaming_message_dump_hdfs_path + added_topic_path
    kafka_data_path = kafka_message_dump_hdfs_path + added_topic_path
        
    # first let's convert hdfs data
    # please note: whether or not to extract_yesterday is needed!
    print("get yesterday string:", extract_yesterday)
    dump_hash_data_to_hdfs(row_hdfs_path, dump_hdfs_path=hash_hdfs_data_path, extract_yesterday=extract_yesterday)
    
    kafka_diff_num, hdfs_diff_num = compare_kafka_and_hdfs(kafka_data_path, 
                                                           hash_hdfs_data_path, 
                                                           extract_yesterday=extract_yesterday,
                                                           save_diff_df=save_diff_df)
    
    print("kakfa_diff_num:", kafka_diff_num)
    print("hdfs_diff_num:", hdfs_diff_num)
    