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


source_path = "/data/insight/cio/alice/jsontoes/full_load__2.1.2.1/20210422"
target_path = "/data/insight/cio/alice/jsontoes/full_load__2.1.2.1/20210422_withclause"

df1 = spark.read.json(source_path)
df2 = spark.read.json(target_path)

# List HDFS paths
from hdfs.ext.kerberos import KerberosClient
import datetime
client = KerberosClient("http://name-node.cioprd.local:50070;http://name-node2.cioprd.local:50070")

latest_full_load_folder_name = "full_load__2.1.2.1"
root_path = "/data/insight/cio/alice/jsontoes/"
folder_list = [x for x in client.list(root_path) if '_withclause' in x]

latest_full_load_time = client.status(os.path.join(root_path, latest_full_load_folder_name))['modificationTime']

latest_full_load_date = datetime.datetime.fromtimestamp(latest_full_load_time/1000)

for x in folder_list:
    try:
        if datetime.datetime.strptime(x.split('/')[-1], '%Y%m%d') >= latest_full_load_date:
            print(x)
    except:
        pass
    


from sklearn.feature_extraction import DictVectorizer

import sklearn

