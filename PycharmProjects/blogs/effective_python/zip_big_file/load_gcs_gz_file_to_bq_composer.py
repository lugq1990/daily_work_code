"""Load GCS GZ file into BQ with composer implement."""

import time
from google.cloud import bigquery
from datetime import timedelta, datetime

from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.contrib.operators.gcs_to_bq import GoogleCloudStorageToBigQueryOperator
from airflow.operators.python_operator import PythonOperator
from google.cloud import storage
from zipfile import ZipFile
from zipfile import is_zipfile
import io
from google.cloud import bigquery
import time


bucket_name = "cloud_sch_test"
dataset_name = 'cloud_load_test'
table_name = 'iris_test'
project_id = "buoyant-sum-302208"
file_name = "out.gz"
# file_name = 'iris_big.zip'


yesterday = datetime.now() - timedelta(days=1)

default_args = {
    'owner': 'lugq',
    'depends_on_past': False,
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': yesterday
}


dag = DAG("Load_file", default_args=default_args, schedule_interval=timedelta(days=1), tags=['load'])

# target field that needed to be configured with pre-defined list or with a JSON file in the GCS, please reference with composer API.
field_list = ['a', 'b', 'c', 'd', 'e', 'name']
schema_fields = [{"name": x, "type": "float"} for x in field_list[:4]]
schema_fields.append({'name': 'e', 'type':'integer'})
schema_fields.append({'name': 'name', 'type':'string'})
# schema_fields = [{"name":x, 'type':'string'} for x in field_list]

gcs_to_bq = GoogleCloudStorageToBigQueryOperator(task_id="gcs_to_bq_directly", 
                bucket=bucket_name, 
                source_objects=[file_name],
                destination_project_dataset_table='{}.{}.{}'.format(project_id, dataset_name, table_name), 
                schema_fields=schema_fields,
                write_disposition='WRITE_TRUNCATE', 
                skip_leading_rows=1,   # skip head
                dag=dag)



