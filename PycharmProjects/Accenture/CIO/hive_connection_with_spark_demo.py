# -*- coding:utf-8 -*-
"""
This is a demo to show how to use Spark to connect to Hive warehouse.

@author: Guangqiang.lu
"""
import os
import sys

# this is to init the env that we would use.
# we do need to init with configuration here.
config = dict()
config["spark_home"] = "/usr/hdp/current/spark2-client"
config["pylib"] = "/python/lib"
config['zip_list'] = ["/py4j-0.10.7-src.zip", "/pyspark.zip"]
# this should be the python env created in Data Science zone or production zone.
config['pyspark_python'] = "/anaconda-efs/sharedfiles/projects/mysched_9376/envs/cap_prd_py36_mml/bin/python"

os.environ["SPARK_HOME"] = config['spark_home']
os.environ["PYSPARK_PYTHON"] = config["pyspark_python"]
os.environ["PYLIB"] = os.environ["SPARK_HOME"] + config['pylib']
zip_list = config['zip_list']
for zip in zip_list:
    sys.path.insert(0, os.environ["PYLIB"] + zip)


# spark-submit --cluster-mode client hive**.py
# we do need to import the whole Spark module here.
from pyspark.sql import SparkSession
from pyspark import SparkConf


# init spark session object, you could do the configuration here with Spark.
conf = SparkConf().setAppName("Myschedule").setMaster("yarn")
# return is a Spark Session object, then you could use this object to do
# some transforms with this object.
spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()

# which database to use?
db_name = "myhrschd_preprod_insight_9376"
# use this database or could be used in later SQL command
spark.sql("use %s" % db_name)

# The SQL logic here.
sql = "select * from tokens_table limit 10"

# execute SQL logic, return is a Spark DataFrame
df = spark.sql(sql)

nums = df.count()

print("Get %d records" % nums)

# pandas DataFrame here in driver memory, then you could do any action with Pandas DataFrame.
pandas_df = df.toPandas()

