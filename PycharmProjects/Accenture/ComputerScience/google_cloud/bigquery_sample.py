# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""
from google.cloud import bigquery

project_id = "sbx-11811-dsd000f-bd-5f8980d7"
client = bigquery.Client(project=project_id)

print(list(client.list_datasets()))

dataset_name = "uccx_11811_raw_test"
table_name = "agent_connection_detail"
dataset = client.get_dataset(dataset_name)
table_ref = dataset.table(table_name)
table = client.get_table(table_ref)
schema = table.schema
schema_names = [s.name for s in schema]
[print(x) for x in schema_names]

query = "select state_name, sales_region from data_first.first_table limit 10"

query_res = client.query(query)

print("Get result:")
res = query_res.result()

# then we could convert the result into a DataFrame
df = res.to_dataframe()



