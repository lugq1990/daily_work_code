# -*- coding:utf-8 -*-
"""This is used to copy the production HDFS satisfied files in HIVE table in the whole files
to another HDFS folder based on the file name last character"""

import os, sys
from hdfs.ext.kerberos import KerberosClient
import pandas as pd
import numpy as np
import tempfile
import logging

logger = logging.getLogger('copy_whole_files')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y%m%d %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)

hdfs_path_part_1 = "/data/insight/cio/alice/contracts_files/whole_files"
hdfs_path_part_2 = "/data/insight/cio/alice/contracts_files/whole_files2"
upload_hdfs_path = "/data/insight/cio/alice/contracts_files/full_loads"
local_path = "/anaconda-efs/sharedfiles/projects/alice_30899/whole_hdfs_files"
hdfs_path_whole = [hdfs_path_part_1, hdfs_path_part_2]

# first init spark
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

logger.info("Init Spark!")
conf = SparkConf().setAppName("create_mapping").setMaster("yarn")
spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()

logger.info("Get HIVE data using spark.")
# Get hive table data using spark
spark.sql("use alice_uat_staging_30899")
hive_df = spark.sql("select concat(document_path_hdfs, '/', documentname, '.txt') as file_name from documents_in_hdfs_vw ").toPandas()

logger.info("Get %d records from HIVE." % (len(hive_df)))

# after get the hive data, then here I should get the whole files from HDFS
client = KerberosClient("http://name-node.cioprd.local:50070")

# first create data_** folder in upload HDFS path
import string
str_list = list(string.digits + string.ascii_uppercase)
folder_name_list = ["data_" + x for x in str_list]

# ensure later step without so many logid, here before to upload the file, should first remove the
# whole upload HDFS folders' files

# logger.info("Start to delete the folders in HDFS.")
# try:
#     for f in folder_name_list:
#         client.delete(os.path.join(upload_hdfs_path, f), recursive=True)
# except Exception as e:
#     logger.error("when to remove the HDFS folder with error: %s" % (e))
#
# # create HDFS data_*** folder
# logger.info("Start to create the data_*** folder in HDFS.")
# try:
#     for f in folder_name_list:
#         client.makedirs(os.path.join(upload_hdfs_path, f))
# except:
#     pass   # just pass as already created



# as the file name from HIVE table is based on HDFS folder, also I could get the file path in HDFS,
# so here I could just download the file to local and upload the files to NEW created folder in HDFS
# Loop for the whole files by download one file and upload one file to the folder according to the last character
tmp_folder = tempfile.mkdtemp()

path_list = hive_df.values.reshape(-1, ).tolist()

# for testing, maybe we run the code many times, so here I don't want to download the same file every time
# so here I just to list the already downloaded files
local_files = os.listdir(local_path)

logger.info("Start to download and upload step.")
for file_path in path_list:
    # first get the file name
    file_name = file_path.split('/')[-1]
    if file_name in local_files:
        continue
    # if "_" in file_name:
    #     last_char = file_name[file_name.index('_') - 1]
    # else:
    #     last_char = file_name[:-4][-1].upper()
    # upload_folder = "data_" + last_char
    # first download this file to tmp folder
    # As I couldn't make the step by step with one file by one file, so here I just download the whole files to
    # local path
    client.download(file_path, os.path.join(local_path, file_name), overwrite=True)
    # after download the file, upload this file,
    # client.upload(local_path=os.path.join(tmp_folder, file_name), hdfs_path=os.path.join(upload_hdfs_path, upload_folder), overwrite=True)
    # # after upload step finish, remove the file in temperate folder
    # os.remove(os.path.join(tmp_folder, file_name))
    if path_list.index(file_path) % 10000 == 0:
        logger.info("Already download %d files" % (path_list.index(file_path)))

logger.info("Download step finished! Start to upload files!")

# first to get the whole files in local directory
already_downloaded_list = os.listdir(local_path)
for file_name in already_downloaded_list:
    if "_" in file_name:
        last_char = file_name[file_name.index('_') - 1].upper()
    else:
        last_char = file_name[-4].upper()
    upload_folder = "data_" + last_char

    client.upload(hdfs_path="/".join([upload_hdfs_path, upload_folder]),
                  local_path=os.path.join(local_path, file_name),
                  overwrite=True,
                  n_threads=3)
    if already_downloaded_list.index(file_name) % 50000 == 0:
        logger.info("Already upload %d files into HDFS." % (already_downloaded_list.index(file_name)))


# if the whole step finished without any error, so for now, we could just remove the whole folder in
# local safely
logger.info("Start to remove the local folder for storing the files.")
import shutil
try:
    shutil.rmtree(local_path)
except Exception as e:
    logger.error("Remove the local folder with error %s " % (e))

logger.info("Whole step finished! You could check the HDFS folder.")






