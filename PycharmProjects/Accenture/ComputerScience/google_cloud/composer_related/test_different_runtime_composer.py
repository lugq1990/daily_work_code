# -*- coding:utf-8 -*-
"""
This is to test different schedule time for composer

@author: Guangqiang.lu
"""
import os
import datetime

import airflow
from airflow import DAG
from airflow.operators.bash_operator import BashOperator


yesterday = datetime.datetime.now() - datetime.timedelta(minutes=3)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': [''],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'start_date': yesterday,
}


with airflow.DAG("new_dag",
                 default_args=default_args,
                 schedule_interval="0-59/2 * * * *") as dag:
    echo = BashOperator(task_id="run_test", bash_command='echo hello')

