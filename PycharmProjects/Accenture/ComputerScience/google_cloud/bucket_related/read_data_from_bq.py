# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""
from google.cloud import bigquery
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'SA-DSD-NPRD_bq.json'

project_id = "npd-65343-datalake-bd-b4f8d566"

client = bigquery.Client(project_id)

print(list(client.list_datasets()))
