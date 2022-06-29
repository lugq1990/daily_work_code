"""Compare kafka incremental data records are equal to HDFS incremental data!

process step: 
    1. get current data num; 
    2. load previous step data info; 
    3. do comparation; 
    4. dump current data num info to disk for next loop compare.
"""

from pyspark.sql import SparkSession
import subprocess
from datetime import datetime
import json

spark = SparkSession.builder.getOrCreate()


topic = "ods-ebs-prod05"
hdfs_folder = "/user/hive/stream/ods-ebs-prod05"

kafka_count_cmd = """/opt/cloudera/parcels/CDH/lib/kafka/bin/kafka-run-class.sh kafka.tools.GetOffsetShell \
  --broker-list IT-Kafka-Node01:9092 \
  --topic {} \
  --time -1 \
  --offsets 1 
"""

data_num_info_dict = {}

def get_kafka_num(topic):
    # get kafka count
    out = subprocess.getoutput(kafka_count_cmd.format(topic))
    # extract total number
    kafka_total_num = sum([int(sub.split(":")[-1]) for sub in out.split('\n') ])
    return kafka_total_num

def get_hdfs_num(hdfs_folder):
    df = spark.read.parquet(hdfs_folder)
    # start to get hdfs folder count
    hdfs_num = df.count()
    return hdfs_num

def get_current_data_num_info():
    # change read spark data first, then kafka 
    hdfs_num = get_hdfs_num(hdfs_folder)
    kafka_total_num = get_kafka_num(topic)
    print("Get hdfs number: {}".format(hdfs_num))
    print("Get kafka number: {}".format(kafka_total_num))
    data_num_info_dict["time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data_num_info_dict["kafka_total_num"] = kafka_total_num
    data_num_info_dict["hdfs_total_num"] = hdfs_num
    data_num_info_dict["historical_total_num"] = kafka_total_num - hdfs_num
    return data_num_info_dict


# todo: should have a functionality to load previous data info and do minus to get increment data number!
def dump_json_to_file(json_obj, file_path="/root/python_env/tmp_data/kafka_num_data.txt"):
    if isinstance(json_obj, dict):
        json_obj = json.dumps(json_obj)
    with open(file_path, "a") as f:
        f.write(json_obj + "\n")

def load_latest_info(file_path="/root/python_env/tmp_data/kafka_num_data.txt"):
    with open(file_path, 'r') as f:
        # only get last line as latest
        data_latest = f.readlines()[-1]
    return json.loads(data_latest)


def compare(current_info_dict, previous_info_dict):
    current_kafka_total_num = current_info_dict.get("kafka_total_num")
    previous_kafka_total_num = previous_info_dict.get("kafka_total_num")
    current_hdfs_total_num = current_info_dict.get("hdfs_total_num")
    previous_hdfs_total_num = previous_info_dict.get("hdfs_total_num")
    kafka_diff = current_kafka_total_num - previous_kafka_total_num
    hdfs_diff = current_hdfs_total_num - previous_hdfs_total_num
    if kafka_diff != hdfs_diff:
        print("At least for now, kafka data is not same with hdfs! Kafka data has {} more then hdfs".format(kafka_diff - hdfs_diff))
        return False
    return True


# as we would more like to run 2 functions in parallel
from multiprocessing import Process, Queue
def get_kafka_num_queue(topic, queue):
    # get kafka count
    out = subprocess.getoutput(kafka_count_cmd.format(topic))
    # extract total number
    kafka_total_num = sum([int(sub.split(":")[-1]) for sub in out.split('\n') ])
    queue.put({"kafka_total_num":kafka_total_num})
    return kafka_total_num


def get_hdfs_num_queue(hdfs_folder, queue):
    df = spark.read.parquet(hdfs_folder)
    # start to get hdfs folder count
    hdfs_num = df.count()
    queue.put({"hdfs_total_num":hdfs_num})
    return hdfs_num



if __name__ == "__main__":
    queue_1 = Queue()
    p1 = Process(target=get_kafka_num_queue, args=(topic, queue_1))
    queue_2 = Queue()
    p2 = Process(target=get_hdfs_num_queue, args=(hdfs_folder, queue_2))
    p1.start()
    p2.start()
    kafka_total_num = queue_1.get()
    hdfs_total_num = queue_2.get()
    print(kafka_total_num, hdfs_total_num)


    # todo: add a functionality at first time to run code, we have to dump current data info into disk!
    current_data_num_info = get_current_data_num_info()
    latest_data_num_info = load_latest_info()
    dump_json_to_file(current_data_num_info)
    compare_res = compare(current_data_num_info, latest_data_num_info)
    print("For topic: {} compare result is : {}".format(topic, str(compare_res)))
