from pyflink.table import TableEnvironment, EnvironmentSettings, TableDescriptor, Schema, DataTypes, FormatDescriptor
from pyflink.common import Row, SimpleStringSchema
from pyflink.table.expressions import lit, col 
from pyflink.table.udf import udtf 
import os
from pyflink.datastream.connectors import kafka
from pyflink.datastream.connectors.kafka import KafkaSource, FlinkKafkaConsumer
from pyflink.datastream import StreamExecutionEnvironment
import json

def dese(x):
    return json.dumps(json.loads(x))

env = StreamExecutionEnvironment.get_execution_environment()

env.add_jars("file:///Users/guangqianglu/Downloads/flink-connector-kafka-3.0.2-1.18.jar")

source = FlinkKafkaConsumer(topics='test', deserialization_schema=SimpleStringSchema(), properties={"bootstrap.servers":'localhost:9092'})

env.add_source(source).map(dese).print()

env.execute()