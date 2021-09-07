# -*- coding:utf-8 -*-
from hdfs.ext.kerberos import KerberosClient
from hdfs import HdfsError
import paramiko
import os
import tempfile
import shutil
import time
import sys
import logging

logger = logging.getLogger('hdfs_to_hive')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y%m%d %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)

start_time = time.time()

host = '10.5.105.51'
port = 22
username = 'ngap.app.alice'
password = 'QWer@#2019'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

ssh.connect(host, port=port, username=username, password=password, look_for_keys=False)

client = KerberosClient("http://name-node.cioprd.local:50070")
client.delete()

hdfs_path = "/data/insight/cio/alice.pp/contracts_files/20190723"
upload_hdfs_path = "/data/insight/cio/alice.pp/contracts_files/test_commond_2"
sftp_path = "/home/ngap.app.alice"

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

conf = SparkConf().setAppName("daily_job_change").setMaster("yarn")
spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()

hive_df = spark.sql("""
Select concat(f.document_path_hdfs,'/',f.documentname,'.txt') as filepath from 
alice_insights_30899.es_metadata_ip e inner join alice_uat_staging_30899.documents_in_hdfs_vw f 
on e.DocumentName = f.documentname where (upper(e.DocumentLanguage)='ENGLISH' or 
upper(e.DocumentLanguage)='N/A') and e.DocumentPath is not null limit 20000
""").toPandas()

file_list = hive_df.values.reshape(-1, ).tolist()


# file_list = client.list(hdfs_path=hdfs_path)

# download files to temperate folder
# tmp_path = "/anaconda-efs/sharedfiles/projects/alice_30899/tmp_files"
tmp_path = tempfile.mkdtemp()

# here is to add the retry logic when the download job fails, then just to retry to download other un-download files
retry_times = 3
satisfy = False
cur_step = 1

while not satisfy:
    t_list = []
    try:
        for f in file_list:
            t_list.append(client.download(local_path=os.path.join(tmp_path, f.split('/')[-1]), hdfs_path=f, overwrite=True))
            if len(t_list) % 1000 == 0:
                print("already donwload %d files. " % (len(t_list)))
        satisfy = True
    except Exception as e:
        # when job fails during download step, first get which files already downloaded,
        # then just change the needed download object file_list
        cur_step += 1
        already_downloaded_list = [x for x in os.listdir(tmp_path) if x.endswith('.txt')]
        file_list = list(set(file_list) - set(already_downloaded_list))
    finally:
        if cur_step > retry_times:
            satisfy = True
            logger.error("When download files for %d times with error %s" % (retry_times, e))
            raise HdfsError("When download files for %d times with error %s" % (retry_times, e))




"""here is just to use gzip file extension to test"""
# gzip_command = "tar cvzf %s/whole.tar.gz %s/*.txt " % (tmp_path, tmp_path)
zip_name = "whole2.zip"
zip_command = "zip -m -q %s %s/*.txt" % (os.path.join(tmp_path, zip_name), tmp_path)
# zip_command = "tar cvzf %s %s/*.txt" % (os.path.join(tmp_path, zip_name), tmp_path)
r = os.system(zip_command)


client.upload(local_path=os.path.join(tmp_path, zip_name), hdfs_path=os.path.join(upload_hdfs_path, zip_name), overwrite=True)

client.delete()
# get zip files size.
print("Zip file size {:.2f} MB.".format(os.path.getsize(os.path.join(tmp_path, zip_name)) >> 20))


# sc = spark.sparkContext
# sc.textFile(os.path.join(upload_hdfs_path, zip_name)).count()
#
# df = spark.read.load(os.path.join(upload_hdfs_path, zip_name))
# df.count()
#
#
# import shutil
# import gzip
#
# down_list = [x for x in os.listdir(tmp_path) if x.endswith('.txt')]
#
# res = ""
# for f in down_list:
#     with open(os.path.join(tmp_path, f), 'r') as f:
#         res += f.read() + '\n'
#
# with gzip.open(os.path.join(tmp_path, zip_name), 'w') as f:
#     f.write(res.encode())
#
# for f in down_list:
#     with open(os.path.join(tmp_path, f), 'rb') as f_in, gzip.open(os.path.join(tmp_path, zip_name), 'wb') as f_out:
#         shutil.copyfileobj(f_in, f_out)






# client.upload(local_path=tmp_path, hdfs_path=upload_hdfs_path + '/')

## recursive upload files one by one
# u_list = []
# for f in file_list:
#     u_list.append(client.upload(local_path=os.path.join(tmp_path, f), hdfs_path=os.path.join(upload_hdfs_path, f), n_threads=4, overwrite=True))
#     if len(u_list) % 100 == 0:
#         print("already put %d files" % (len(u_list)))

# not work, it will put the local folder to HDFS
# client.upload(local_path=tmp_path, hdfs_path=upload_hdfs_path, n_threads=4)

# client.delete(upload_hdfs_path, recursive=True)


put_command = "hdfs dfs -put -f %s/*.txt %s" % (tmp_path, upload_hdfs_path)
# As I notice one thing that for the current solution, I have already download the files
# to local server folder, so here shouldn't use the paramiko to execute the command,
# should just use os.system to execute the command to put local files to HDFS


### As sometimes with os.system to put files to HDFS maybe with some unexpected error,
# here just add the logic to re-put the files to HDFS if os fails by using the client to put files
satisfy_upload = False
up_step = 0

os_common_res = os.system(put_command)
# os.system will return 1 when job fails
if os_common_res == 0:
    satisfy_upload = True

# if os fails, just use client to put
while not satisfy_upload:
    # first to get how many files already been putted.
    already_put_list = [x for x in client.list(upload_hdfs_path) if x.endswith('.txt')]
    un_putted_list = list(set([x.split('/')[-1] for x in file_list]) - set(already_put_list))
    if len(un_putted_list) != 0 and up_step < retry_times:
        try:
            # loop for the not_putted file list
            for file_name in un_putted_list:
                client.upload(local_path=os.path.join(tmp_path, file_name),
                              hdfs_path=os.path.join(upload_hdfs_path, file_name),
                              overwrite=True)
        except Exception as e:
            up_step += 1
            logger.warning("Already upload files with %d times." % (up_step))
    elif len(un_putted_list) == 0 or up_step > retry_times:
        satisfy_upload = True


#
# stdin, stdout, stderr = ssh.exec_command(put_command)
# print(stdout.readlines())

t = [x for x in client.list(upload_hdfs_path) if x.endswith('.txt')]

assert len(file_list) == len(t)

shutil.rmtree(tmp_path)

end_time = time.time()

print("Whole step for {0:d} files use {1:.2f} seconds.".format(len(file_list), (end_time - start_time)))


client.delete(upload_hdfs_path, recursive=True)
client.makedirs(upload_hdfs_path)





