"""
Load file from GCS to BQ with CSV file type using `Cloud functions`.
"""
from google.cloud import storage
from zipfile import ZipFile
from zipfile import is_zipfile
import io
from google.cloud import bigquery
import time

default_file_name = 'iris.zip'



# def unzip_file(file_name='iris.zip'):
#     client = storage.Client()

#     bucket_name = "cloud_sch_test"

#     bucket = client.get_bucket(bucket_name)
#     blob = bucket.blob(file_name)

#     zipbytes = io.BytesIO(blob.download_as_string())

#     if is_zipfile(zipbytes):
#         with ZipFile(zipbytes, 'r') as my_zip:
#             for content_file_name in my_zip.namelist():
#                 print("Now to process file: {}".format(content_file_name))
#                 # Read the content of the file as string, in fact is bytes
#                 content = my_zip.read(content_file_name)
#                 # Upload the content of the file into GCS with String IO
#                 upload_blob = bucket.blob("iris_python.gz")
#                 upload_blob.upload_from_string(content)


def load_gz_file_to_bq(file_name='iris_python.csv'):
    client = bigquery.Client()

    bucket_name = "cloud_sch_test"
    dataset_name = 'cloud_load_test'
    table_name = 'iris_test'
    project_id = "buoyant-sum-302208"

    file_gcs_path = 'gs://{}/{}'.format(bucket_name, file_name)

    columns_list = list('abcde')
    schema_list = [bigquery.SchemaField(x, 'STRING') for x in columns_list]


    job_config = bigquery.LoadJobConfig(schema=schema_list, source_format=bigquery.SourceFormat.CSV, skip_leading_rows=1, autodetect=True)


    dataset = client.dataset(dataset_name)
    table_id = dataset.table(table_name)
    table_id = "{}.{}.{}".format(project_id, dataset_name, table_name)

    start_time = time.time()
    load_job = client.load_table_from_uri(file_gcs_path, table_id, job_config=job_config)

    load_job.result()

    print("Load action takes:{} seconds.".format(time.time() - start_time))


    table = client.get_table(table_id)
    print("Already loaded {} rows.".format(table.num_rows))


# Load .gz file from GCS to Big Query directly. If with .CSV file, then just change the parameter from .gz to .csv
def hello_gcs(event, context):
    """Triggered by a change to a Cloud Storage bucket.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    file_name = event['name']
    
    print("Get file name: {} from event.".format(file_name))

    if file_name.endswith(".csv"):
        # unzip_file(file_name)
        load_gz_file_to_bq(file_name)
        print("File {} process has finished".format(file_name))
    else:
        print("WARNING: Get file:{}".format(file_name))




from google.cloud import bigquery

client = bigquery.Client()

bucket_name = "cloud_sch_test"
dataset_name = 'sample'
table_name = 'test_quota'
project_id = "buoyant-sum-302208"
file_name = "sample.gz"

file_gcs_path = 'gs://{}/{}'.format(bucket_name, file_name)

columns_list = list('a')
schema_list = [bigquery.SchemaField(x, 'STRING') for x in columns_list]


job_config = bigquery.LoadJobConfig(schema=schema_list, source_format=bigquery.SourceFormat.CSV, skip_leading_rows=1, autodetect=True)


dataset = client.dataset(dataset_name)
table_id = dataset.table(table_name)
table_id = "{}.{}.{}".format(project_id, dataset_name, table_name)

start_time = time.time()
load_job = client.load_table_from_uri(file_gcs_path, table_id, job_config=job_config)

load_job.result()

print("Load action takes:{} seconds.".format(time.time() - start_time))
