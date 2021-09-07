# -*- coding:utf-8 -*-
"""
This script is used to convert the ipynb into some format file that supported by jupyter.

@author: Guangqiang.lu
"""
import os
import shutil
import logging
import warnings

warnings.simplefilter('ignore')
logger = logging.getLogger(__file__)


base_path = "C:/Users/guangqiiang.lu/Documents/lugq/workings/notebooks_Python/GCP_tutorial"

# currently is just `html`
support_extensions = ['html']


def convert_to_extension():
    # first we should get whole notebook files, just get name without extension
    nb_files = [x.split('.')[0] for x in os.listdir(base_path) if x.endswith('ipynb')]

    for extension in support_extensions:
        # first let's check whether extension folder has been created
        extension_path = os.path.join(base_path, extension)
        if not os.path.exists(extension_path):
            # if not exist, we should first try to create one
            try:
                os.mkdir(extension_path)
                logger.info("Folder: {} has been created!".format(extension))
            except IOError as e:
                raise IOError("When try to create folder: {}, get error:{}".format(extension, e))

        # then we should get the currently file names in extension folder
        extension_files = [x.split('.')[0] for x in os.listdir(extension_path) if x.endswith(extension)]

        # Get files names haven't been converted.
        not_convert_files = list(set(nb_files) - set(extension_files))
        # add .ipynb to the file names
        not_convert_files = [x + '.ipynb' for x in not_convert_files]

        # as currently jupyter doesn't support with set destination path,
        # so here just to copy the files from source to destination
        logger.info("Now to copy the files to {} folder.".format(extension))
        for file in not_convert_files:
            shutil.copy(os.path.join(base_path, file), os.path.join(extension_path, file))
        logger.info("Files has been copied.")

        # then we could start our convert on haven't converted
        for file in not_convert_files:
            try:
                # so here just to change source path from base to extension path.
                src_path = os.path.join(extension_path, file)
                os.system("jupyter nbconvert {} --to {}".format(src_path, extension))
                logger.info("Converted file: {} to extension: {} finished".format(file, extension))
            except Exception as e:
                logger.error("When to convert file: {} to {} get error: {}".format(file, extension, e))

        # here we do need to remove the source files in destination folder
        logger.info("Now to remove destination source file.")
        for file in not_convert_files:
            try:
                os.remove(os.path.join(extension_path, file))
            except Exception as e:
                logger.error("When to remove folder: {} source file get error: {}".format(extension, e))


    logger.info("Whole steps finished.")


if __name__ == '__main__':
    convert_to_extension()

import datetime

import airflow
from airflow.operators import bash_operator
from airflow.operators.python_operator import PythonOperator
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

YESTERDAY = datetime.datetime.now() - datetime.timedelta(days=1)

default_args = {
    'owner': 'Composer Example',
    'depends_on_past': False,
    'email': [''],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'start_date': YESTERDAY,
}

dag = airflow.DAG("training_lr", "catchup=False", default_args=default_args, schedule_interval=datetime.timedelta(days=1))

x, y = load_iris(return_X_y=True)
lr = LogisticRegression()

def train_model():
    print("Start to train model")
    lr.fit(x, y)

    score = lr.score(x, y)
    print("Model test score: {}".format(score))

PythonOperator(dag=dag,
               task_id='my_task_powered_by_python',
               provide_context=False,
               python_callable=train_model)

print("Whole training finished.")