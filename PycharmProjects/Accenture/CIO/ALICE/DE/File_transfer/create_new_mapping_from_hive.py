# -*- coding:utf-8 -*-
"""This is to create a new mapping file in production env.
There are 4 steps: 1. init spark in MML; 2. Get matadata from HIVE; 3. create the new mapping file;
4. put the new mapping to HDFS production env"""
import os
import sys
import pandas as pd
from hdfs.ext.kerberos import KerberosClient
import datetime
import shutil
import tempfile
import logging

logger = logging.getLogger('create_new_mapping')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y%m%d %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)

"""Here I don't have to write these logic to function, as this code will just be used for once!"""

### init parameters
hdfs_parent_path = '/data/insight/cio/alice/contracts_files'
whole_folder_1 = "whole_files"
whole_folder_2 = "whole_files2"
hdfs_upload_path = '/data/insight/cio/alice.pp/hivetable/documents_name_new'


logger.info("Start to init env and create spark instance.")

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


logger.info("Get Hive data using Spark.")
# this will be changed in the future.
# spark.sql("use alice_uat_staging_30899")

# get data from hive table and convert it to a pandas DataFrame for later step process,
# even if the data will be loaded to the master client, so no other ways to do it.
# here this logic could be changed without just get the result using one table.

# hive_df = spark.sql("select documentname as full_path, split(documentname, '/')[5] as name from es_metadata_ip").toPandas()
## this is based on the hive logic to get the document path and document name
hive_df = spark.sql("""
SELECT concat(substring(mmrtopdir,6,length(trim(mmrtopdir))),regexp_extract(trim( documentpath),'([^.]+)\.[^.]*$', 1) ) as documentpath
,documentname
, upper(substr(documentpath, locate('.', documentpath, 1) + 1, length(documentpath))) as extension
FROM (
SELECT documentcategory
,documenteffectivedate
,documenttitle
,documentpath
,documentname
,documentid
,documentlanguage
,documentexpirationdate
,mmrtopdir
,masterclientname
,clientleader
,clientcreateddate
,clientname
,contractleader
,contractcreateddate
,contractname
,engagementenddate
,clientservicegroup
,clientserviceregion
,clientcountry
,contractcountry
,accessgroups
,lastactiondatetime
,count(1) OVER (PARTITION BY documentname) AS countname
FROM (
SELECT trim(cd.doccategory) AS documentcategory
,trim(cd.effectivedt) AS documenteffectivedate
,trim(cd.doctitle) AS documenttitle
,trim(cd.doclocation) AS documentpath
,regexp_extract(trim(cd.doclocation), '([^/.]+)\.[^.]*$', 1) AS documentname
,cd.docmmruri AS documentid
,trim(cd.doclanguage) AS documentlanguage
,trim(cd.expirationdt) AS documentexpirationdate
,trim(cd.mmrtopdir) AS mmrtopdir
,trim(cli.mstrclientname) AS masterclientname
,trim(cli.clientlead) AS clientleader
,cli.clientcreatedt AS clientcreateddate
,trim(cli.clientname) AS clientname
,trim(co.contractlead) AS contractleader
,trim(co.contractcreatedt) AS contractcreateddate
,trim(co.contractname) AS contractname
,trim(co.contractenddt) AS engagementenddate
,trim(co.clientservicegroup) AS clientservicegroup
,trim(gr.georegiondesc) AS clientserviceregion
,trim(clic.countryname) AS clientcountry
,trim(coc.countryname) AS contractcountry
,da.accessgroups AS accessgroups
,cd.lastactiondatetime AS lastactiondatetime
,row_number() OVER (
PARTITION BY cd.docmmruri ORDER BY cd.lastactiondatetime DESC
) AS rankuri
FROM alice_insights_30899.documents cd
LEFT JOIN (
SELECT *
FROM (
SELECT clienturi
,countryid
,mstrclientname
,clientlead
,clientcreatedt
,clientname
,row_number() OVER (
PARTITION BY clienturi ORDER BY lastactiondatetime DESC
) AS rn
FROM alice_insights_30899.clients
) aa
WHERE aa.rn = 1
) cli ON cd.clienturi = cli.clienturi
LEFT JOIN (
SELECT *
FROM (
SELECT contractlead
,contractcreatedt
,contractname
,contractenddt
,clientservicegroup
,contracturi
,georegionid
,countryid
,row_number() OVER (
PARTITION BY contracturi ORDER BY lastactiondatetime DESC
) AS rn
FROM alice_insights_30899.contracts
) bb
WHERE bb.rn = 1
) co ON cd.contracturi = co.contracturi
LEFT JOIN alice_insights_30899.geographicregion gr ON co.georegionid = gr.georegionid
LEFT JOIN alice_insights_30899.countries clic ON clic.countryid = cli.countryid
LEFT JOIN alice_insights_30899.countries coc ON coc.countryid = co.countryid
LEFT JOIN alice_insights_30899.accessgroup da ON cd.docmmruri = da.docmmruri
) query
WHERE query.rankuri = 1
) es
WHERE es.countName = 1
""").toPandas()

hive_df = hive_df[hive_df['extension'] != 'ZIP']

hive_df.columns = ['full_path', 'name', 'extension']
hive_df['extension'] = hive_df['extension'].apply(lambda x: x.upper())

logger.info("Get %d records from HIVE" % (len(hive_df)))

logger.info("Get HDFS files DataFrame.")
# here is to list the whole files in HDFS, here should be the whole files in HDFS
folder_list = []
folder_list.append(os.path.join(hdfs_parent_path, whole_folder_1))
folder_list.append(os.path.join(hdfs_parent_path, whole_folder_2))

client = KerberosClient("http://name-node.cioprd.local:50070")

# in fact, here couldn't just get the file name, as we also need the path that the file locates,
# here should also save the path, in fact, there could use the dictionary to save the types,
# but I would just save two objects that save different contents
file_list = []
locate_list = []
for folder in folder_list:
    files = client.list(folder)  # this is a list
    file_list.extend(files)
    locate_list.extend([folder] * len(files))

# as mapping data doesn't have the extension, here should just remove the extension of the HDFS file list
file_list = [x[:-4] for x in file_list]

# create a DataFrame for the file list for later step join
hdfs_df = pd.DataFrame(file_list, columns=['file_name'])
hdfs_df['file_locate'] = locate_list

logger.info("Get %d records in DHFS." % (len(hdfs_df)))

# merge this two dataframe
hive_df.rename(index=str, columns={'name': 'file_name'}, inplace=True)

logger.info("Merge two DataFrame and save it to local temperate folder.")
merged_df = pd.merge(hdfs_df, hive_df, how='inner', on='file_name')

merged_df = merged_df[['file_name', 'file_locate', 'full_path', 'extension']]

logger.info("Get Merged dataframe with %d records" % (len(merged_df)))

# as the document_in_hdfs also with date column, here is just add with current date
merged_df['date'] = datetime.datetime.now().strftime('%Y%m%d')

### here I don't need to create the mapping file to SFTP, just make the document_in_hdfs files
### and put this file to HDFS, also overwrite this file!!!

# first to save file to local temperate folder, and put this file to HDFS, after this, just remove this folder
tmp_folder = tempfile.mkdtemp()
local_file_path = os.path.join(tmp_folder, "whole_file.csv")

merged_df = merged_df[['file_name', 'file_locate', 'full_path', 'date', 'extension']]
merged_df.to_csv(local_file_path, header=False, index=False)

logger.info("Start to put file to HIVE external HDFS path with overwriting.")
client.upload(hdfs_path=hdfs_upload_path, local_path=local_file_path, overwrite=True)

# after upload step finished, then just remove the temperate folder and file
try:
    logger.info("Remove the temperate folder in local server.")
    shutil.rmtree(tmp_folder)
except IOError as e:
    logger.error("Fail to remove the temperate folder in local folder.")




from sklearn.linear_model import LogisticRegression