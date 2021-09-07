# -*- coding:utf-8 -*-
"""
This is a demonstration of how to use google composer with apache airflow

@author: Guangqiang.lu
"""
import datetime

import airflow
from airflow.operators import bash_operator

yesterday = datetime.datetime.now() - datetime.timedelta(days=1)

default_args = {
    'owner': 'Composer Example',
    'depends_on_past': False,
    'email': [''],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'start_date': yesterday,
}

with airflow.DAG('composer_sample_dag', 'catchup=false',
                 default_args=default_args,
                 schedule_interval=datetime.timedelta(days=1)) as dag:
    print_dag_run_conf = bash_operator.BashOperator(task_id='print_dag_run_conf', bash_command='echo {{dag_run.id}}')
