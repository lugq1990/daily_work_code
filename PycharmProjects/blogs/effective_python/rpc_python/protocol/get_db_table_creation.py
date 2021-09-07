import os
import sys


config = dict()
config["spark_home"] = "/usr/hdp/current/spark2-client"
config["pylib"] = "/python/lib"
config['zip_list'] = ["/py4j-0.10.7-src.zip", "/pyspark.zip"]
config['pyspark_python'] = "/anaconda-efs/sharedfiles/projects/alice_30899/envs/smart_legal_pipeline_DS/bin/python"

os.system("kinit -k -t /etc/security/keytabs/ngap.app.alice.keytab ngap.app.alice")
os.environ["SPARK_HOME"] = config['spark_home']
os.environ["PYSPARK_PYTHON"] = config["pyspark_python"]
os.environ["PYLIB"] = os.environ["SPARK_HOME"] + config['pylib']
zip_list = config['zip_list']
for zip in zip_list:
    sys.path.insert(0, os.environ["PYLIB"] + zip)

# This module must be imported after environment init.
from pyspark.sql import SparkSession
from pyspark import SparkConf

conf = SparkConf().setAppName("create_mapping").setMaster("yarn")
spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()


