# -*- coding:utf-8 -*-
"""This is to implement the logic to get the whole files content and combined with mapping file
to create the json files in HDFS"""

import logging
import os, sys
from hdfs.ext.kerberos import KerberosClient
import tempfile
import shutil

logger = logging.getLogger('create_new_mapping')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y%m%d %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)

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

spark.sql("use alice_uat_staging_30899")

hive_df = spark.sql("select concat(document_path_hdfs, '/', documentname, '.txt') as file_name from documents_in_hdfs_vw limit 600 ").toPandas()

spark.sql("").toJSON()
client = KerberosClient("http://name-node.cioprd.local:50070")

tmp_folder = tempfile.mkdtemp()

upload_hdfs_path = ["/data/insight/cio/alice.pp/contracts_files/whole_files", "/data/insight/cio/alice.pp/contracts_files/whole_files2"]

file_list = hive_df.values.reshape(-1, ).tolist()
for file_path in file_list:
    file_name = file_path.split('/')[-1]
    client.download(hdfs_path=file_path, local_path=os.path.join(tmp_folder, file_name))
    if file_list.index(file_path) < 300:
        client.upload(local_path=os.path.join(tmp_folder, file_name), hdfs_path=upload_hdfs_path[0], overwrite=True)
    else:
        client.upload(local_path=os.path.join(tmp_folder, file_name), hdfs_path=upload_hdfs_path[1], overwrite=True)

print("upload step finished! remove the temperate folder.")
try:
    shutil.rmtree(tmp_folder)
except Exception as e:
    pass


"""This is just to get the normal files name but could't be combine with hive table"""
hive_df = spark.sql("select documentname as file_name from documents_in_hdfs_vw ").toPandas()
hive_list = hive_df.values.reshape(-1,).tolist()
hive_list = [x + '.txt' for x in hive_list]

whole_hdfs_folder = ["/data/insight/cio/alice/contracts_files/whole_files", "/data/insight/cio/alice/contracts_files/whole_files2"]
file_list = []
for f in whole_hdfs_folder:
    file_list.append(client.list(f))

whole_file_part_one = file_list[0]
whole_file_part_two = file_list[1]

# get common file name
common_list = list(set(whole_file_part_one).intersection(set(whole_file_part_two)))

whole_list = []
whole_list.extend(whole_file_part_one)
whole_list.extend(whole_file_part_two)

diff_list = list(set(whole_list) - set(hive_list))
diff_list = [x.replace('.txt', '') for x in diff_list]

import pandas as pd
import tempfile
tmp_folder = tempfile.mkdtemp()

df = pd.DataFrame(diff_list, columns=['file_name'])
df.to_csv(os.path.join(tmp_folder, 'diff_files.csv'), header=False, index=False)
diff_files_in_meta_hdfs = "/data/insight/cio/alice.pp/diff_files_tmp"
client.upload(hdfs_path=diff_files_in_meta_hdfs, local_path=os.path.join(tmp_folder, 'diff_files.csv'), overwrite=True)

df = spark.sql()
df.write.json

