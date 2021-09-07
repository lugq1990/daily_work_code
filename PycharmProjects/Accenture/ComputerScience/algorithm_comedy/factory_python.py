
from google.cloud import bigquery

project_name = "loyal-weaver-296802"
dataset_name = "cloud_dataset"
table_name = "iris"
bucket_name = "gcs_to_bq_cloud_func_new"
file_name = "data.csv"

# where is the file in bucket
file_uri = "gs://{}/{}".format(bucket_name, file_name)
# where to load the data into BQ
dataset_table_name = "{}.{}.{}".format(project_name, dataset_name, table_name)

# BQ client
client = bigquery.Client()

columns_name = ['a', 'b', 'c', 'd', 'label']
schema_list = [bigquery.SchemaField(x, 'float') for x in columns_name]


def create_dataset_and_table():
    """
    Create dataset and table
    """
    # create dataset and table
    try:
        dataset = client.create_dataset(dataset_name)
        print("dataset has been created!")
    except Exception as e:
        print("When try to create dataset: {} with error:{}".format(dataset_name, e))
        pass

    dataset = client.get_dataset(dataset_name)
    table_ref = dataset.table(table_name)
    table = bigquery.Table(table_ref, schema=schema_list)

    # create table
    try:
        table = client.create_table(table, exists_ok=True)
        print("table:{} has been created.".format(table_name))
    except Exception as e:
        print("When create table with error:{}".format(e))


def load_gcs_file_into_bq_new(event, context):
    """
    Based on the pubsub topic to trigger cloud function to load
    files in bucket into bigquery directly.
    """
    # first to create dataset and table
    create_dataset_and_table()

    # config job config, this is whole thing that we need to config
    # during the loading step: where the file and where to put data,
    # and the file information, how to load the file: `append` or `truncate`
    job_config = bigquery.LoadJobConfig(
        schema=schema_list,
        skip_leading_rows=1,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        source_format=bigquery.SourceFormat.CSV,
        field_delimiter=','
    )

    load_job = client.load_table_from_uri(file_uri, dataset_table_name, job_config=job_config)

    # wait to finish
    load_job.result()

    print("Load action has finished without error")

    print("Whole step finished.")



