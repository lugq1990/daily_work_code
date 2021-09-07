from google.cloud import bigquery
client = bigquery.Client()
dataset_ref = client.dataset("uccx_india_11811_raw")
table_ref = dataset_ref.table("team")

 

job_config = bigquery.LoadJobConfig(
            skip_leading_rows=1,
            source_format=bigquery.SourceFormat.CSV,
            field_delimiter='|',
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            encoding='utf-8',
            autodetect=True,
            allow_quoted_newlines=True
        )
# uri = "gs://india_test_contact/ContactCallDetail_2021-01-18-205143.csv"
uri = "gs://prd-65343-datalake-bd-22042-dsd-ca-dsdgcs-raw/Team_india.csv"


target_table = "prd-65343-datalake-bd-88394358.uccx_india_11811_raw.team"

# gcs -> df -> gcs file

load_job = client.load_table_from_uri(uri, target_table, job_config=job_config)
load_job.result() 
