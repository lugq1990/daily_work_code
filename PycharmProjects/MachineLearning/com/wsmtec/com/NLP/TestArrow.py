# -*- coding:utf-8 -*-
import pandas as pd
from pyspark.sql import SparkSession
import time
import sys
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='process data')
    parser.add_argument('--is_enable', type=str, help='need enable arrow', required=True)
    parser.add_argument('--nums', type=str, help='how many samples', required=True)

    args = vars(parser.parse_args())

    is_enable = args['is_enable']
    nums = args['nums']

    print('Get parameter ', is_enable, nums)

    spark = SparkSession.builder.enableHiveSupport().getOrCreate()


    spark.conf.set('spark.sql.execution.arrow.enabled',is_enable)

    # spk = spark.enableHiveSupport().getOrCreate()
    # spark.config('spark.sql.execution.arrow.enabled','True')

    df = spark.sql('select * from etl.bds_madrid_tb_orderdetails_d limit '+nums)

    st = time.time()
    data = df.toPandas()
    print('total seconds ',(time.time()-st))
    print('parameter is ', is_enable, 'and type is ',type(is_enable))

    st2 = time.time()
    new_data = spark.createDataFrame(data)

    new_data.createOrReplaceTempView('t')
    r = spark.sql('select * from t')
    print('inverse time is ', (time.time() - st2))
    print('the args:',args)