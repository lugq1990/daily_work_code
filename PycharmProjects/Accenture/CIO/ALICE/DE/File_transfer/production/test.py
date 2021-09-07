# -*- coding:utf-8 -*-
from hdfs.ext.kerberos import KerberosClient
import paramiko
import os

host = '10.5.105.51'
port = 22
username = 'ngap.app.alice'
password = 'QWer@#2019'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

ssh.connect(host, port=port, username=username, password=password, look_for_keys=False)

client = KerberosClient("http://name-node.cioprd.local:50070")

hdfs_path = "/data/insight/cio/alice.pp/contracts_files/20190723"
upload_hdfs_path = "/data/insight/cio/alice.pp/contracts_files/test_commond"
sftp_path = "/home/ngap.app.alice"

file_list = client.list(hdfs_path=hdfs_path)

res = ""
for f in file_list:
    res += f + '\n'

sftp = ssh.open_sftp()

with sftp.open(os.path.join(sftp_path, 'file.txt'), 'w') as f:
    f.write(res)



move_file_command = """
    awk '{print $1}' %s | while read num
    do
    hdfs dfs -cp %s/$num.txt %s
    done 
    """ % (os.path.join(sftp_path, 'file.txt'), hdfs_path, upload_hdfs_path)

stin, stout, sterr = ssh.exec_command(move_file_command)
stout.readlines()

"""
    awk '{print $1}' /home/ngap.app.alice/file.txt  | while read num
    do
    hdfs dfs -cp -f /data/insight/cio/alice.pp/contracts_files/20190723/$num /data/insight/cio/alice.pp/contracts_files/test_commond
    done 
"""


"""This is to test to download the files to temperate folder and use the command to put *.txt file to 
needed folder in HDFS"""
from hdfs.ext.kerberos import KerberosClient
import paramiko
import os
import tempfile
import shutil
import time
import sys

start_time = time.time()

host = '10.5.105.51'
port = 22
username = 'ngap.app.alice'
password = 'QWer@#2019'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

ssh.connect(host, port=port, username=username, password=password, look_for_keys=False)

client = KerberosClient("http://name-node.cioprd.local:50070")

hdfs_path = "/data/insight/cio/alice.pp/contracts_files/whole_files"
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

conf = SparkConf().setAppName("create_mapping").setMaster("yarn")
spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()

hive_df = spark.sql("""
Select concat(f.document_path_hdfs,'/',f.documentname,'.txt') as filepath from 
alice_uat_staging_30899.es_metadata_ip e inner join alice_uat_staging_30899.documents_in_hdfs_vw f 
on e.DocumentName = f.mapping_full_path where (upper(e.DocumentLanguage)='ENGLISH' or 
upper(e.DocumentLanguage)='N/A') and e.DocumentPath is not null limit 10
""").toPandas()

file_list = hive_df.values.reshape(-1, ).tolist()


# file_list = client.list(hdfs_path=hdfs_path)

# download files to temperate folder
# tmp_path = "/anaconda-efs/sharedfiles/projects/alice_30899/tmp_files"
tmp_path = tempfile.mkdtemp()


t_list = []
for f in file_list:
    t_list.append(client.download(local_path=os.path.join(tmp_path, f), hdfs_path=os.path.join(hdfs_path, f)))
    if len(t_list) % 500 == 0:
        print("already donwload %d files. " % (len(t_list)))

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
os.system(put_command)

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





"""This is to get how many duplicate files in document_in_hdfs"""
import os
from hdfs.ext.kerberos import KerberosClient
import numpy as np

local_path = '/mrsprd_data/Users/ngap.app.alice/shell_test/mapping_shell/tmp_data'
hdfs_path = "/data/insight/cio/alice/hivetable/documents_name/"

client = KerberosClient("http://name-node.cioprd.local:50070")

# first to get the whole files in HDFS
file_list = client.list(hdfs_path=hdfs_path)

# get the whole files content with one dataframe
res_list = []
for f in file_list:
    with client.read(os.path.join(hdfs_path, f)) as reader:
        res_list.append(reader.read())

out_list = []
for data in res_list:
    data = data.decode('utf-8')
    data = data.split('\n')
    data = [x.split(',') for x in data]
    out_list.extend(data)

out_list[0]

out_arr = np.empty((len(out_list), len(out_list[0])), dtype=str)
for i in range(len(out_list)):
    out_arr[i] = out_list[i]

out_arr = np.asarray(out_list)

import pandas as pd

hive_df = pd.DataFrame(out_list, columns=['file_name', 'hdfs_path', 'full_path', 'dt', 'extension'])

local_df = pd.read_csv(os.path.join(local_path, os.listdir(local_path)[0]))
local_df.columns = ['id', 'full_path', 'other']

common_df = pd.merge(hive_df, local_df, how='inner', on='full_path')

not_common_df = pd.merge(local_df, hive_df, how='left', left_on='full_path')

not_common_list = list(set(local_df['full_path'].values) - set(hive_df['full_path'].values))

not_common_df = pd.DataFrame(not_common_list, columns=['full_path'])

not_common_df = pd.merge(local_df, not_common_df, on='full_path', how='inner')

upload_hdfs_path = '/data/insight/cio/alice/lugq'

import tempfile

tmp_path = tempfile.mkdtemp()
not_common_df.to_csv(os.path.join(tmp_path, 'not_common.csv'), header=True, index=False)
client.upload(hdfs_path=upload_hdfs_path, local_path=os.path.join(tmp_path, 'not_common.csv'))




"""This is just to test how much time the SQL run finished"""
import time
import sys
import os

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


start_time = time.time()
sql = """
SELECT docmmruri
,docrecordid
,doctitle
,docentrydt
,mmrexternalid
,clienturi
,contracturi
,docfilename
,docextn
,docformat
,doclocation
,nbrofpages
,doccategory
,docdospath
,coalesce(max(CASE 
WHEN dfieldname = 'Accenture Signee'
THEN dfieldval
END), 'n/a') AS acnsignee
,coalesce(max(CASE 
WHEN dfieldname = 'Client Signee'
THEN dfieldval
END), 'n/a') AS clientsignee
,coalesce(max(CASE 
WHEN dfieldname = 'Effective Date'
THEN CONCAT (
substring(dfieldval, 1, 4)
,'-'
,substring(dfieldval, 5, 2)
,'-'
,substring(dfieldval, 7, 2)
)
END), 'n/a') AS effectivedt
,coalesce(max(CASE 
WHEN dfieldname = 'Expiration Date'
THEN CONCAT (
substring(dfieldval, 1, 4)
,'-'
,substring(dfieldval, 5, 2)
,'-'
,substring(dfieldval, 7, 2)
)
END), 'n/a') AS expirationdt
,coalesce(max(CASE 
WHEN dfieldname = 'Language'
THEN dfieldval
END), 'n/a') AS doclanguage
,CASE lower(docdospath)
WHEN '%global%\\'
THEN 'HPRM_DocStore/P1/1'
WHEN '\\\\vrtva25071\\hprm_docstore1\\'
THEN 'HPRM_DocStore2/P1/2'
WHEN '\\\\vrtva25071\\hprm_docstore2\\'
THEN 'HPRM_DocStore3/P1/3'
WHEN '\\\\vrtva25072\\hprm_docstore2\\'
THEN 'HPRM_DocStore3/P1/3'
ELSE 'Data Not Expected'
END mmrtopdir
,CASE lower(docdospath)
WHEN '%global%\\'
THEN '/alice_data/HPRM_DocStore/P1/1'
WHEN '\\\\vrtva25071\\hprm_docstore2\\'
THEN '/alice_data/HPRM_DocStore2/P1/2'
ELSE 'N/A'
END alicetopdir
,lastactiondatetime
,current_timestamp() dataloadtime
,from_unixtime(unix_timestamp(CURRENT_DATE (), 'yyyy-mm-dd'), 'yyyyMMdd') AS dt
FROM (
SELECT rc.uri AS docmmruri
,rc.recordid AS docrecordid
,rc.title AS doctitle
,CASE rc.regdatetime
WHEN NULL
THEN NULL
ELSE rc.regdatetime
END AS docentrydt
,rc.externalid AS mmrexternalid
,coalesce(mc.uri, c.uri) AS clienturi
,e.uri AS contracturi
,el.refilename AS docfilename
,el.reextension AS docextn
,el.remimetype AS docformat
,CONCAT (
'/'
,regexp_replace(el.resid, '\\+', '\\/')
) AS doclocation
,el.renbrpages AS nbrofpages
,fp.fpterm AS doccategory
,sto.esdospathlong AS docdospath
,df.exfieldname AS dfieldname
,ds.usvfieldval AS dfieldval
,rc.lastactiondatetime AS lastactiondatetime
FROM alice_insights_30899.alice_hrpmtables_001 rc
LEFT OUTER JOIN alice_insights_30899.alice_hrpmtables_001 e ON rc.rccontaineruri = e.uri
AND e.rcrectypeuri = 4
LEFT OUTER JOIN alice_insights_30899.alice_hrpmtables_001 c ON e.rccontaineruri = c.uri
AND c.rcrectypeuri = 3
LEFT OUTER JOIN alice_insights_30899.alice_hrpmtables_001 mc ON rc.rccontaineruri = mc.uri
AND mc.rcrectypeuri = 3
LEFT OUTER JOIN mmr_4339.tsrecelec el ON el.uri = rc.uri
LEFT OUTER JOIN mmr_4339.tselecstor sto ON sto.uri = el.restoreuri
LEFT OUTER JOIN mmr_4339.tsfileplan fp ON rc.rcfileplanuri = fp.uri
LEFT OUTER JOIN alice_insights_30899.alice_hrpmtables_002 ds ON ds.usvobjecturi = rc.uri
LEFT OUTER JOIN mmr_4339.tsexfield df ON ds.usvfielduri = df.uri
AND df.uri IN (
31
,32
,601
)
WHERE rc.rcrectypeuri = '7'

UNION

SELECT rc.uri AS docmmruri
,rc.recordid AS docrecordid
,rc.title AS doctitle
,CASE rc.regdatetime
WHEN NULL
THEN NULL
ELSE rc.regdatetime
END AS docentrydt
,rc.externalid AS mmrexternalid
,coalesce(mc.uri, c.uri) AS clienturi
,e.uri AS contracturi
,el.refilename AS docfilename
,el.reextension AS docextn
,el.remimetype AS docformat
,CONCAT (
'/'
,regexp_replace(el.resid, '\\+', '\\/')
) AS doclocation
,el.renbrpages AS nbrofpages
,fp.fpterm AS doccategory
,sto.esdospathlong AS docdospath
,df.exfieldname AS dfieldname
,ddt.udvfieldval AS dfieldval
,rc.lastactiondatetime AS lastactiondatetime
FROM alice_insights_30899.alice_hrpmtables_001 rc
LEFT OUTER JOIN alice_insights_30899.alice_hrpmtables_001 mc ON rc.rccontaineruri = mc.uri
AND mc.rcrectypeuri = 3
LEFT OUTER JOIN alice_insights_30899.alice_hrpmtables_001 e ON rc.rccontaineruri = e.uri
AND e.rcrectypeuri = 4
LEFT OUTER JOIN alice_insights_30899.alice_hrpmtables_001 c ON e.rccontaineruri = c.uri
AND c.rcrectypeuri = 3
LEFT OUTER JOIN mmr_4339.tsrecelec el ON el.uri = rc.uri
LEFT OUTER JOIN mmr_4339.tselecstor sto ON sto.uri = el.restoreuri
LEFT OUTER JOIN mmr_4339.tsfileplan fp ON rc.rcfileplanuri = fp.uri
LEFT OUTER JOIN mmr_4339.tsudfdatev ddt ON ddt.udvobjecturi = rc.uri
LEFT OUTER JOIN mmr_4339.tsexfield df ON ddt.udvfielduri = df.uri
AND df.uri IN (
33
,34
)
WHERE rc.rcrectypeuri = '7'
) mmrqry
GROUP BY docmmruri
,docrecordid
,doctitle
,docentrydt
,mmrexternalid
,clienturi
,contracturi
,docfilename
,docextn
,docformat
,doclocation
,nbrofpages
,doccategory
,docdospath
,lastactiondatetime
"""

df = spark.sql(sqlQuery=sql)
end_time = time.time()

print("Whole SQL execute with %.2f minutes" % (end_time - start_time)/60)





"""This is to test with doc files with how many file could be joined with mapping file!"""
import boto3
import os
import sys


access_key = 'AKIAJGK5GH7CU46OSS5Q'
secret_key = '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'
bucket_name = 'aliceportal-30899-prod'
# s3_folder_name = 'Delta'
s3_folder_name = 'Delta_Missing_Null'

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

sql = "select distinct(documentname) as documentname from es_metadata_ip_fullload"
# sql = "select documentname from es_metadata_ip_duplicate"
spark.sql("use alice_insights_30899")
hive_df = spark.sql(sql).toPandas()

session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
client = session.client('s3')
s3 = session.resource('s3')
my_bucket = s3.Bucket(bucket_name)

file = b''
for f in my_bucket.objects.filter(Prefix=s3_folder_name + '/'):
    f_key = f.key
    if f_key.endswith('.txt') and f_key.split('/')[1].lower().startswith('mapping'):
        print('Get mapping file', f_key)
        file += client.get_object(Bucket=bucket_name, Key=f_key)['Body'].read()

# convert byte to string
import pandas as pd
data = file.decode()
data = [x.split('\t') for x in data.split('\n')]
s3_df = pd.DataFrame(data, columns=['name', 'documentname', 'dt'])
common_df = pd.merge(hive_df, s3_df, on='documentname', how='inner')

print("There are %d files not in HIVE!" % (len(s3_df) - len(common_df)))

a = pd.DataFrame(pd.unique(hive_df['documentname']), columns=['documentname'])




# this is to get the difference parts
l_list = list(set(s3_df['documentname']) - set(common_df['documentname']))

not_common_df = pd.DataFrame(l_list, columns=['documentname'])
not_common_df = pd.DataFrame(not_common_df['documentname'].apply(lambda x: x.split('/')[-1]))

convert_df = spark.sql("select * from (select doclanguage, clienturi, split(doclocation, '/')[3] as f from  documents)t where f is not NULL").toPandas()
convert_df['documentname'] = convert_df['f'].apply(lambda x: x.split('.')[0])

not_common_df_with_lang = pd.merge(convert_df, not_common_df, on='documentname')

unique_lang = pd.unique(not_common_df_with_lang['doclanguage'])
unique_uri = pd.unique(not_common_df_with_lang['clienturi'])
print('unique language: ', unique_lang)
print('client uri unique: ', unique_uri)
print('unique language number: ', len(unique_lang))
print('unique uri number: ', len(unique_uri))







"""This is to test to download HDFS json file to load and zip many of them to be one"""
from hdfs.ext.kerberos import KerberosClient
import os

client = KerberosClient("http://name-node.cioprd.local:50070;http://name-node2.cioprd.local:50070")

local_path = '/anaconda-efs/sharedfiles/projects/alice_30899/full_load_json'
hdfs_path = '/data/insight/cio/alice/jsontoes/full_load_withclause/'

file_list = [x for x in client.list(hdfs_path) if x.endswith('.json')]
file_list.sort()

n_folder = 4
folder_prefix = 'part_'
# first should create n_folder local folder

try:
    for i in range(n_folder):
        os.mkdir(os.path.join(local_path, folder_prefix + str(i)))
except:
    pass

folder_list = os.listdir(local_path)
folder_list.sort()

folder_files_list = [os.listdir(os.path.join(local_path, x)) for x in folder_list]

# then here should download HDFS files to that folder
each_folder_samples = int(len(file_list) / n_folder)

# in case the job fails, here should store the already download files list
# also add with retry logic
sati = False
while not sati:
    try:
        for i in range(n_folder):
            down_folder = os.path.join(local_path, folder_prefix + str(i))
            print("download files to %s " % down_folder)
            j = 0
            # each folder should just with len(file_list)/ n_folder
            for f in file_list[i * each_folder_samples: (i + 1) * each_folder_samples]:
                if f in folder_files_list[i]:
                    continue
                client.download(hdfs_path=os.path.join(hdfs_path, f), local_path=os.path.join(down_folder, f), overwrite=True, n_threads=0)
                j += 1
                if j % 50 == 0:
                    print("For folder %s, already download %d files." % (down_folder, j))
        sati = True
    except:
        print("download file with error, retry")

# then should do zip command to make the whole folder to be one zip file
zip_command = 'zip -r %s %s'
for folder in folder_list:
    exe_command = zip_command % (os.path.join(local_path, folder + '.zip'), '/'.join([local_path, folder, '*.json']))
    print("Now to execute command: %s" % exe_command)
    os.system(exe_command)




folder_files_list = [os.listdir(os.path.join(local_path, x)) for x in folder_list]
already_download_list = []
for i in range(len(folder_files_list)):
    already_download_list.extend(folder_files_list[i])

file_list = client.list(hdfs_path)

not_download_list = list(set(file_list) - set(already_download_list))
for f in not_download_list:
    client.download(hdfs_path=os.path.join(hdfs_path, f), local_path='/'.join([local_path, 'part_0', f]))




host = '18.210.190.6'
port = 22
username = 'guangqiang.lu'
password = 'Lgq1990!'
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

ssh.connect(host, port=port, username=username, password=password, look_for_keys=False)
sftp = ssh.open_sftp()


# """This is to change date columns for files in HDFS"""
# from hdfs.ext.kerberos import KerberosClient
# import os
# import tempfile
# import datetime
# import pandas as pd
#
# client = KerberosClient("http://name-node.cioprd.local:50070;http://name-node2.cioprd.local:50070")
#
# tmp_path = tempfile.mkdtemp()
#
# hdfs_path = '/data/insight/cio/alice.pp/hivetable/documents_name'
# hdfs_org_path = '/data/insight/cio/alice.pp/hivetable/documents_name'
#
# date_list = ['20190101', '20190102', '20190103', '20190104']
#
# df_list = []
#
# for i in range(len(date_list)):
#     file_name = date_list[i] + '.csv'
#     print('now to download %s' % file_name)
#     client.download(hdfs_path=os.path.join(hdfs_org_path, file_name), local_path=os.path.join(tmp_path, file_name), overwrite=True)
#     df = pd.read_csv(os.path.join(tmp_path, file_name), header=None)
#     cur_date = (datetime.datetime.now() + datetime.timedelta(days=i + 1)).strftime('%Y%m%d')
#     df.iloc[:, 3] = cur_date
#     df.to_csv(os.path.join(tmp_path, file_name), index=False, header=False)
#     print("start to upload file %s" % file_name)
#     client.upload(hdfs_path=os.path.join(hdfs_path, file_name), local_path=os.path.join(tmp_path, file_name), overwrite=True)




from hdfs.ext.kerberos import KerberosClient
import pandas as pd
import os

cur_path = os.path.abspath(os.curdir)
file_name = [x for x in os.listdir(cur_path) if x.endswith('.csv')][0]

hdfs_path = '/data/insight/cio/alice/contracts_files'
catch_path_list = ['20190101', '20190102', '20190103', '20190104']
whole_path = ['whole_files', 'whole_files2']

client = KerberosClient("http://name-node.cioprd.local:50070;http://name-node2.cioprd.local:50070")

catch_list = []
for f in catch_path_list:
    catch_list.extend(client.list(os.path.join(hdfs_path, f)))

whole_list = []
for f in whole_path:
    whole_list.extend(client.list(os.path.join(hdfs_path, f)))

other_list = list(set(whole_list) - set(catch_list))

new_df = pd.read_csv(os.path.join(cur_path, file_name))
new_df.columns = ['a', 'documentpath']
new_list = new_df['documentpath'].values.tolist()

new_list = [x.strip().split('/')[-1].split('.')[0] + '.txt' for x in new_list]

in_catch_list = list(set(catch_list).intersection(set(new_list)))
print('How many in catch: %d' % len(in_catch_list))
in_whole_list = list(set(whole_list).intersection(set(new_list)))
print('How many in whole: %d' % len(in_whole_list))




"""This is to check the files in which mapping file."""
import time
import sys
import os

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
sql = """
select docmmruri , documentpath, pp.documentname from documents dd 
inner join es_metadata_ip pp on dd.docmmruri=pp.documentid
inner join (select distinct regexp_extract(mapping_full_path, '([^/])[^_]+', 0) documentname from documents_in_hdfs) doc on pp.documentname = doc.documentname
where upper(pp.documentlanguage) IN ('ENGLISH','N/A') 
and upper(substr(documentpath, locate('.', documentpath, 1) + 1, length(documentpath))) in 
('7Z','HTM','HTML','MPP','MSG','P7M','PPT','PPTX','RAR','RTF','TXT','XLS','XLSB','XLSM','XLSX','XPS') 
"""

need_df = spark.sql(sql).toPandas()

name_list = need_df['documentname'].values.tolist()

from hdfs.ext.kerberos import KerberosClient
import pandas as pd
import os

client = KerberosClient("http://name-node.cioprd.local:50070;http://name-node2.cioprd.local:50070")

hdfs_path = '/data/insight/cio/alice/contracts_files/mapping_files'

file_list = client.list(hdfs_path)
file_list.sort(reverse=True)

import tempfile
tmp_path = tempfile.mkdtemp()

for f in file_list:
    client.download(hdfs_path=os.path.join(hdfs_path, f), local_path=os.path.join(tmp_path, f))

locate_dic = dict()

for f in file_list:
    with open(os.path.join(tmp_path, f), 'r') as fr:
        data = [x.replace('\n', '') for x in fr.readlines()]
        data = [x.split('\t')[1] for x in data]
        for file in name_list:
            if file in data:
                print("Get file: %s in mapping file: %s" % (file, f))
                locate_dic[file] = f



# This is get the whole files in MMR S3 bucket files list
import boto3
import os
import pandas as pd

# this should be changed to local folder.
local_path = os.getcwd()

secret_key = 'Z4d6tlLDRipLWmR43calDvE2SgxHagHNQa8ZBaV9'
access_key = 'AKIAJMR43CLDBRZMVLAA'
bucket_name = '4339-mmr-prod'

session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
client = session.client('s3')
s3 = session.resource('s3')
my_bucket = s3.Bucket(bucket_name)

folder_list = ['DocStore', 'DocStore1', 'DocStore2']

file_list = []

for folder in folder_list:
    current_folder_list = []
    # loop for each folder to get the file
    for file in my_bucket.objects.filter(Prefix=folder + '/'):
        if '.' not in file.key:
            continue
        current_folder_list.append(file.key)
        if len(current_folder_list) % 10000 == 0:
            print("For folder %s have get %d files" % (folder, len(current_folder_list)))
    file_list.extend(current_folder_list)


file_df = pd.DataFrame(file_list, columns=['file_name'])
file_df['file_path'] = file_df['file_name'].apply(lambda x: x.split('.')[0])
file_df['extension'] = file_df['file_name'].apply(lambda x: x.split('.')[-1])
file_df.drop(['file_name'], axis=1, inplace=True)

file_df.to_csv(os.path.join(tmp_path, 'whole_file_in_mmr.csv'), header=False, index=False)



"""This is to make 9*9 matrix"""
for i in range(1, 10):
    current_string = ''
    for j in range(1, 10):
        # if i value is greater than j value, then we could just add the
        # i * j value
        if i >= j:
            current_string = current_string + ' ' + str(i * j)
    print(current_string)


import os
import tempfile
from hdfs.ext.kerberos import KerberosClient

tmp_path = tempfile.mkdtemp()
client = KerberosClient("http://name-node.cioprd.local:50070;http://name-node2.cioprd.local:50070")

with open(os.path.join(tmp_path, 'res.txt'), 'w') as f:
    f.write("this is something could be added.")

hdfs_path = '/data/raw/mysched'
os.listdir(tmp_path)
client.upload(hdfs_path=os.path.join(hdfs_path, 'res.txt'), local_path=os.path.join(tmp_path, 'res.txt'))
