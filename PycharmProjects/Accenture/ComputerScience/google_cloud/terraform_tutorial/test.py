import os
import datetime

import airflow
from airflow import DAG
from airflow.contrib.operators.gcs_to_bq import GoogleCloudStorageToBigQueryOperator
from airflow.contrib.operators.bigquery_operator import BigQueryOperator
from airflow.operators.bash_operator import BashOperator

from google.cloud import storage

# first try to get the file list
project_id = "sbx-11811-dsd-bd-e33b324b"

# Currently we couldn't use storage client, as the SA doesn't have the permission
# client = storage.Client(project_id)

# bucket_name = "talend_first_test"
# bucket = client.get_bucket(bucket_name)

# file_list = [x.name for x in bucket.list_blobs() if x.name.startswith("Mapping_File") and x.name.endswith('csv')]
# file_list = ["Mapping_File_Call_Disposition.csv"]
file_list = ['Mapping_File/Call_Disposition.csv',
             'Mapping_File/Call_Disposition_Flag.csv',
             'Mapping_File/GMT_Offset_Matrix.csv',
             'Mapping_File/Peripheral_Call_Type.csv',
             'Mapping_File/Queue_Names.csv']

sql = """
select
cd.calldisposition as calldisposition ,
"need_to_change" as calldispositionname ,
FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S', CURRENT_TIMESTAMP()) AS createdttm, 
FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S', CURRENT_TIMESTAMP())AS updatedttm, 
SESSION_USER() AS createuserid, 
SESSION_USER() AS updateuserid from 
CCCI_22042.Call_Disposition cd 
where cd.calldisposition is not null"""
sql = sql.replace('\n', ' ')

file_name_list = [x.split('/')[1].split('.')[0] for x in file_list]

yesterday = datetime.datetime.now() - datetime.timedelta(days=1)

default_args = {
    'owner': 'lugq',
    'depends_on_past': False,
    'email': [''],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'start_date': yesterday,
}

# These configurations are used for load file into bigquery
bucket_name = "first_talend_job"
dataset_name = "CCCI_22042"
source_file_list = ["{}".format(x) for x in file_list]

destination_table_list = [dataset_name + "." + x.split('/')[1].replace('.csv', '') for x in file_list]

with airflow.DAG("new_dag",
                 default_args=default_args,
                 schedule_interval=datetime.timedelta(days=1)) as dag:
    for i in range(len(file_list)):
        gcs_to_bq = GoogleCloudStorageToBigQueryOperator(task_id="load_data_to_bq_{}".format(file_name_list[i]),
                                                         bucket=bucket_name,
                                                         source_objects=[source_file_list[i]],
                                                         destination_project_dataset_table=destination_table_list[i],
                                                         write_disposition='WRITE_TRUNCATE',
                                                         skip_leading_rows=1)
    # Excute SQL logic
    dest_dataset_name = "CCCI_22042_Landed"
    dest_table_name = "Call_Disposition"
    bq_command = BigQueryOperator(task_id="execute_sql",
                                  sql=sql,
                                  destination_dataset_table='{}.{}'.format(dest_dataset_name, dest_table_name),
                                  write_disposition = 'WRITE_TRUNCATE',
                                  use_legacy_sql=False)
    gcs_to_bq >> bq_command


# This is for test to get table schema into a file
from google.cloud import bigquery
client = bigquery.Client()

dataset = [x.dataset_id for x in client.list_datasets()]

res_list = []

for d in dataset:
    table_list = [x.table_id for x in client.list_tables(d)]
    table_schem_list = []
    for table_id in table_list:
        table_ref = client.get_dataset(d).table(table_id)
        table = client.get_table(table_ref)
        schema_list = [','.join([d, table_id, x.name, x.field_type]) for x in table.schema]
        table_schema_str = '\n'.join(schema_list)
        table_schem_list.append(table_schema_str)
    res_list.append('\n'.join(table_schem_list))

with open("table_info.txt", 'w') as f:
    f.write('\n'.join(res_list))

import os

print(os.listdir())

with open('table_info.txt', 'r') as f:
    data = f.readlines()


# Bellow is the functionality to load the txt file and make it into a csv file
import pandas as pd
with open('table_info.txt', 'r') as f:
    data = f.readlines()

data_new = [x.replace('\n', '').split(',') for x in data]

df = pd.DataFrame(data_new, columns=['dataset', 'table', 'schema', 'type'])
df.to_csv('dataset_info.csv', index=False)
