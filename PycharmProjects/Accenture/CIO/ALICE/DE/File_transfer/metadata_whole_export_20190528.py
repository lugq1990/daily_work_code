# -*- coding:utf-8 -*-
import os
from pyspark.sql import SparkSession
import datetime
import logging

logger = logging.getLogger('metadata')


file_path = '/sftp/cio.alice/metadata/metadata_new'
# file_path = 'C:/Users/guangqiiang.lu/Documents/lugq/workings/201904/dataTransfer'
file_name = 'new_sql.sql'

sql_list = []
# cause production python version is 2.6, doesn't support encoding, here should io open
from io import open
with open(os.path.join(file_path, file_name), 'r', encoding="utf-8") as f:
    sql_list = f.readlines()

# after get whole query, just get the command and query title
title_list = []
command_list = []
for f in sql_list:
    # cause there are also '=' in command, here couldn't just make the command with '=' as split,
    #
    title_list.append(f.split('=')[0].split('_')[0])
    start_select_index = f.index('select')
    command = f[start_select_index:]
    command_list.append(command)

print(command_list[5])

# cause the SQL command is just insert overwrite to HDFS, here should just use Spark to execute the command
spark = SparkSession.builder.enableHiveSupport().getOrCreate()

# after get the query, then here could just use the spark to make the implement to execute SQL
hdfs_path = 'file://///sftp/cio.alice/SQLData/metadata/'
# hdfs_path = '/data/insight/cio/alice/sftp_data/'

# Here I will make the code to run with the datetime folder
try:
    date_str = datetime.datetime.now().strftime('%Y%m%d')
    hdfs_path_date = os.path.join(hdfs_path, date_str)
    os.mkdir(hdfs_path_date)
except Exception as e:
    pass

table1 = "Alice_MasterClient.csv"
table2 = "Alice_Client.csv"
table3 = "Alice_ClientLeader.csv"
table4 = "Alice_ClientRegion.csv"
table5 = "Alice_ClientCountry.csv"
table6 = "Alice_ClientServiceGroup.csv"
table7 = "Alice_Contract.csv"
table8 = "Alice_ContractLeader.csv"
table9 = "Alice_ContractCountry.csv"
table10 = "Alice_Document.csv"
table11 = "Alice_Mapping_Contract.csv"

def execute_function(sql, table):
    hdfs_path_new = hdfs_path_date + table
    df = spark.sql(sql)
    print('Now is for table:', format(table))
    if table != table10:
        df.coalesce(1).write.format("com.databricks.spark.csv").options(delimiter=";").mode("overwrite").option(
            "header", "False").save(hdfs_path_new)
    else:
        df.repartition(1).write.format("com.databricks.spark.csv").options(delimiter=";").mode("overwrite").option(
            "header", "False").save(hdfs_path_new)

table_list = []
table_list.append(table1)
table_list.append(table2)
table_list.append(table3)
table_list.append(table4)
table_list.append(table5)
table_list.append(table6)
table_list.append(table7)
table_list.append(table8)
table_list.append(table9)
table_list.append(table10)
table_list.append(table11)

# here is to start run the spark code
for sql, table in zip(command_list, table_list):
    execute_function(sql, table)

# after the whole step finish, I will use the os command to run the copy command, after the whole step finish,
# then remove the .csv folder in the SFTP
for table in table_list:
    os.system('cp %s/*.csv %s'% (os.path.join(hdfs_path[hdfs_path_date.index('/sftp'):], date_str+table), hdfs_path[hdfs_path.index('/sftp'):] + table.split('.')[0] + '.txt'))
    # os.system('rm -rf {}'.format(os.path.join(hdfs_path_date, table)))

print('Whole step finished!')

