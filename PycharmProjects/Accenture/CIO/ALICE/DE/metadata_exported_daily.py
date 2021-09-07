"""This is used to make the preproduction data file transfer"""
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

### This function is used to make dataframe to local file in production env
# df.coalesce(1).write.format("com.databricks.spark.csv").options(delimiter="Ð").mode("overwrite").option("header","False").save("file:////sftp/cio.alice.pp/metadata/t.csv")
hdfs_path = 'file:////sftp/cio.alice/metadata/metadata_new/'
# hdfs_path = '/data/insight/cio/alice/sftp_data/'

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
# FIRST just to use database
spark.sql('use alice_uat_staging_30899')
# This is for table1
sql_table_1 = "select if(mstrclientnbr is null, -99999, mstrclientnbr) as MasterClientID, mstrclientname as MasterClientName, current_date() as UpdateDate from clients group by mstrclientnbr, mstrclientname"
hdfs_path_1 = hdfs_path + table1
df_table_1 = spark.sql(sql_table_1)
df_table_1.coalesce(1).write.format("com.databricks.spark.csv").options(delimiter="Ð").mode("overwrite").option("header","False").save(hdfs_path_1)

sql_table_2 = "select if(clienturi is null, -99999, clienturi) as clienturi, clientname, current_date() as dt from clients group by clienturi, clientname union select -99999 as clienturi, '' as clientname, current_date() as dt"
hdfs_path_2 = hdfs_path + table2
df_table_2 = spark.sql(sql_table_2)
df_table_2.coalesce(1).write.format("com.databricks.spark.csv").options(delimiter="Ð").mode("overwrite").option("header","False").save(hdfs_path_2)


sql_table_3 = "select if(clientleaduri is null,-99999, clientleaduri) as clientleaduri, clientlead,current_date()  as  dt from clients group by clientleaduri, clientlead"
hdfs_path_3 = hdfs_path + table3
df_table_3 = spark.sql(sql_table_3)
df_table_3.coalesce(1).write.format("com.databricks.spark.csv").options(delimiter="Ð").mode("overwrite").option("header","False").save(hdfs_path_3)

sql_table_4 = "select if(georegionid is null, -99999, georegionid) as georegionid, georegiondesc, current_date() as dt from geographicregion group by georegionid, georegiondesc union select -99999 as georegionid, '' as georegiondesc, current_date() as dt"
hdfs_path_4 = hdfs_path + table4
df_table_4 = spark.sql(sql_table_4)
df_table_4.coalesce(1).write.format("com.databricks.spark.csv").options(delimiter="Ð").mode("overwrite").option("header","False").save(hdfs_path_4)


sql_table_5 = "select if(countryid is null, -99999, countryid) as countryid,  clientcountry, current_date() as dt from clients group by countryid, clientcountry union select -99999 as countryid, '' as clientcountry, current_date() as dt"
hdfs_path_5 = hdfs_path + table5
df_table_5 = spark.sql(sql_table_5)
df_table_5.coalesce(1).write.format("com.databricks.spark.csv").options(delimiter="Ð").mode("overwrite").option("header","False").save(hdfs_path_5)


sql_table_6 = "select if(clientservicegroup is null, '-99999', clientservicegroup) as clientservicegroup,  current_date() as dt from contracts group by clientservicegroup union select '-99999' as clientservicegroup,  current_date() as dt"
hdfs_path_6 = hdfs_path + table6
df_table_6 = spark.sql(sql_table_6)
df_table_6.coalesce(1).write.format("com.databricks.spark.csv").options(delimiter="Ð").mode("overwrite").option("header","False").save(hdfs_path_6)


sql_table_7 = "select if(contracturi is null, -99999, contracturi) as contracturi, contractname, current_date() as dt from contracts where dt = regexp_replace(current_date(), '-', '') group by contracturi, contractname union select -99999 as contracturi, '' as contractname, current_date() as dt"
hdfs_path_7 = hdfs_path + table7
df_table_7 = spark.sql(sql_table_7)
df_table_7.coalesce(1).write.format("com.databricks.spark.csv").options(delimiter="Ð").mode("overwrite").option("header","False").save(hdfs_path_7)


sql_table_8 = "select if(contractleaduri is null, -99999, contractleaduri) as contractleaduri,  contractlead, current_date() as dt from contracts where contractleaduri is not null group by contractleaduri, contractlead union select -99999 as contractleaduri,  '' as contractlead, current_date() as dt"
hdfs_path_8 = hdfs_path + table8
df_table_8 = spark.sql(sql_table_8)
df_table_8.coalesce(1).write.format("com.databricks.spark.csv").options(delimiter="Ð").mode("overwrite").option("header","False").save(hdfs_path_8)

sql_table_9 = "select if(countryid is null, -99999, countryid) as countryid, countryname, current_date() as dt from countries group by countryid, countryname union select -99999 as countryid, '' as countryname, current_date() as dt"
hdfs_path_9 = hdfs_path + table9
df_table_9 = spark.sql(sql_table_9)
df_table_9.coalesce(1).write.format("com.databricks.spark.csv").options(delimiter="Ð").mode("overwrite").option("header","False").save(hdfs_path_9)


sql_table_10 = """
select doc1.docmmruri as docmmruri, doc1.doccategory as doccategory, doc1.doclanguage as doclanguage, doc1.dt as dt from
(select if(docmmruri is null, -99999, docmmruri) as docmmruri, doccategory as doccategory, 
 doclanguage as doclanguage,  current_date() as dt,
 regexp_extract(doclocation, '([^/.]+)\.[^.]*$',1) as documentname 
 from documents where upper(doclanguage) in ('ENGLISH','N/A') and doclocation is not null and dt = regexp_replace(current_date(), '-', '')
 ) doc1,
 (select documentname from full_name) doc2
 where  doc1.documentname = doc2.documentname
 union select -99999 as docmmruri, '' as doccategory, '' as doclanguage, current_date() as  dt
 """
hdfs_path_10 = hdfs_path + table10
df_table_10 = spark.sql(sql_table_10)
df_table_10.repartition(1).write.format("com.databricks.spark.csv").options(delimiter="Ð").mode("overwrite").option("header","False").save(hdfs_path_10)


sql_table_11 = """
select
if(mstrclientnbr is null, -99999, mstrclientnbr) as mstrclientnbr, 
if(clienturi is null, -99999, clienturi) as clienturi, 
if(clientleaduri is null, -99999, clientleaduri) as clientleaduri,
if(georegionid is null, -99999, georegionid) as georegionid, 
if(ClientCountryID is null, -99999, ClientCountryID) as ClientCountryID, 
if(clientservicegroup is null, -99999, clientservicegroup) as clientservicegroup, 
if(contracturi is null, -99999, contracturi) as contracturi, 
if(contractleaduri is null, -99999, contractleaduri) as contractleaduri, 
if(ContractCountryID is null, -99999, ContractCountryID) as ContractCountryID, 
if(docmmruri is null, -99999, docmmruri) as docmmruri, 
current_date() as dt 
 from 
(select t2.mstrclientnbr as mstrclientnbr, 
t1.clienturi as clienturi,
 t2.clientleaduri as clientleaduri, 
 t3.georegionid as georegionid, 
 t2.countryid as ClientCountryID, 
t3.clientservicegroup as clientservicegroup,
 t1.contracturi as contracturi, 
 t3.contractleaduri as contractleaduri, 
 t3.countryid as ContractCountryID, 
 t1.docmmruri as docmmruri, 
 t1.dt as dt 
from (
select doc1.clienturi as clienturi, doc1.contracturi as contracturi, doc1.docmmruri as docmmruri, doc1.dt as dt from
(select clienturi, contracturi,docmmruri,dt, documentname   from 
(select if(clienturi is null, -99999, clienturi) as clienturi, 
if(contracturi is null, -99999, contracturi) as contracturi,  
if(docmmruri is null, -99999, docmmruri) as docmmruri, dt,
regexp_extract(doclocation, '([^/.]+)\.[^.]*$',1) as documentname 
from documents
where upper(doclanguage) in ('ENGLISH','N/A') and doclocation is not null and  dt = regexp_replace(current_date(), '-', '')
)doc
 group by clienturi, contracturi, docmmruri, dt, documentname) doc1,
 (select documentname from full_name) doc2
 where doc1.documentname = doc2.documentname
) t1 
left join (
select if(mstrclientnbr is null, -99999, mstrclientnbr) as mstrclientnbr, 
if(clientleaduri is null, -99999, clientleaduri) as clientleaduri, 
if(countryid is null, -99999, countryid) as countryid, 
if(clienturi is null, -99999, clienturi) as clienturi 
from clients group by clienturi,clientleaduri, mstrclientnbr, countryid) t2
on t1.clienturi = t2.clienturi
left join (select if(georegionid is null, -99999, georegionid) as georegionid, 
if(contractleaduri is null, -99999, contractleaduri) as contractleaduri, 
if(countryid is null, -99999, countryid) as countryid, 
 if(contracturi is null, -99999, contracturi) as contracturi,
  if(clientservicegroup is null, -99999, clientservicegroup) as clientservicegroup 
  from  contracts group by contracturi,countryid, georegionid,contractleaduri, clientservicegroup)t3 
on t1.contracturi = t3.contracturi)t4
"""
hdfs_path_11 = hdfs_path + table11
df_table_11 = spark.sql(sql_table_11)
df_table_11.coalesce(1).write.format("com.databricks.spark.csv").options(delimiter="Ð").mode("overwrite").option("header","False").save(hdfs_path_11)



# HERE Because of the Spark write out file is a folder, I have to run some command to transfer the csv file to txt file
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

# Copy all the csv file to txt file using command to do this
# import os
# for t in table_list:
#     os.system('cp /sftp/cio.alice.pp/metadata/%s/*.csv /sftp/cio.alice.pp/metadata/%s.txt'%(t, t[:-4]))

import os
for t in table_list:
    os.system("cp -f /sftp/cio.alice/metadata/metadata_new/%s/*.csv /sftp/cio.alice/metadata/metadata_new/%s.txt"%(t, t[:-4]))
    # os.system("hdfs dfs -copyToLocal /data/insight/cio/alice/sftp_data/%s/*.csv /sftp/cio.alice/metadata/%s.txt"%(t, t[:-4]))
