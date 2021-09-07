# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
import datetime

# this is global variables
is_first_time = False
read_table = 'dm.shopping_behavior'
mul_write_table = 'dm.multi_label_res'
res_write_table = 'dm.user_regression_score'

def enable_arrow(init=True):
    if init:
        spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    else:
        pass

# model_path = '/home/luguangqiang/pyspark/recommendation/model_path'
# model_path = './py2env_pick.zip/py2env_pick/lib/python2.7/site-packages/taobao_libs/models'

# In case the created multi spark, make a method for singleton spark
def get_single_spark():
    if 'spark' not in globals():
        globals()['spark'] = SparkSession.builder.appName('Online_TF').enableHiveSupport().getOrCreate()
    return globals()['spark']


def print_sql(sql):
    print('Now executing SQL :')
    print(sql)

# This function is used to compute the multi-label task and sum-score task
def multi_label_classification_sum_regression(model_path =None, online=False, online_data=None, stat_date=None):
    if online:
        if online_data is None:
            raise ValueError('If this is online, online_data is must provided!')

        data = online_data.toPandas()

        if data.shape[1] != 13:
            raise ValueError('Online data columns length must be 13! Given is %d.'%(data.shape[1]))
    else:
        query_sql = "select mobile,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12 from(" \
                    "select mobile, month_no,sum(recharge) as t1, sum(food) as t2, " \
                    "sum(commodity) as t3, sum(clothing) as t4, sum(tools) as t5, " \
                    "sum(elect) as t6,sum(publish) as t7, sum(entert) as t8, " \
                    "sum(equip) as t9, sum(noness) as t10, sum(medicine) as t11, " \
                    "sum(other) as t12 from %s group by mobile, month_no)t"%read_table
        print_sql(query_sql)

        data = spark.sql(query_sql).toPandas()


    # before we running the multi-score task, we first run sum-score problem
    score_pred_task(pred_data=data, online=online, online_data=online_data, stat_date=stat_date)
    print('The sum-score task is finished! Then run the multi-label task.')

    pred_data = np.array(data.iloc[:, 1:]).reshape(-1, 12, 12)
    mobile = pd.DataFrame(data['mobile'].unique(), columns=['mobile'])

    # load the already trained model
    model = tf.keras.models.load_model(model_path + '/LSTM_multi_class.h5')
    pred = model.predict(pred_data)
    f = lambda x:1 if x>=.5 else 0
    pred = pd.DataFrame(pred).applymap(f)

    # return pred is DataFrame that means for whether or not the indexed score column appeared
    out = pd.concat((mobile, pred), axis=1)

    res = spark.createDataFrame(out)
    # res = res.withColumn('stat_date', lit(stat_date))

    res.createOrReplaceTempView('multi')

    # insert the result to Hive table
    # here just get model data, if this is not null, then we need not to create table
    if(is_first_time):
        #spark.sql("drop table %s "%mul_write_table)
        spark.sql("create table %s(mobile string, rechg_0 int, rechg_1 int, rechg_2 int, "
                  "rechg_3 int, foo_0 int, foo_1 int, foo_2 int, foo_3 int, comdy_0 int,"
                  " comdy_1 int, comdy_2 int, comdy_3 int, clo_0 int, clo_1 int, clo_2 int, "
                  "clo_3 int, tols_0 int, tols_1 int, tols_2 int, tols_3 int, tols_4 int, "
                  "ele_0 int, ele_1 int, ele_2 int, ele_3 int, pub_0 int, pub_1 int, "
                  "pub_2 int, pub_3 int, ent_0 int, ent_1 int, ent_2 int, ent_3 int, "
                  "eqp_0 int, eqp_1 int, eqp_2 int, eqp_3 int, eqp_4 int, ness_0 int, "
                  "ness_1 int, ness_2 int, ness_3 int, med_0 int, med_1 int, med_2 int, "
                  "med_3 int, med_4 int, oth_0 int, oth_1 int, oth_2 int, oth_3 int) "
                  "partitioned by (stat_date string) stored as orc"%mul_write_table)

    in_query = "insert overwrite table %s partition (stat_date=\'%s') select * from multi"%(mul_write_table, stat_date)
    print_sql(in_query)

    spark.sql(in_query)
    print('All finished')

    # all done, release the memory
    spark.catalog.dropTempView('multi')


# This function is used to predict the sum-score for each person, using LSTM model to predict
def score_pred_task(pred_data=None, online=False, online_data=None, stat_date=None):
    # because the data for sum-score is same with multi-label, we can re-use it.
    if pred_data is None:
        data = spark.sql("select mobile,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12 from("
                         "select mobile, month_no,sum(recharge) as t1, sum(food) as t2, "
                         "sum(commodity) as t3, sum(clothing) as t4, sum(tools) as t5, "
                         "sum(elect) as t6,sum(publish) as t7, sum(entert) as t8, "
                         "sum(equip) as t9, sum(noness) as t10, sum(medicine) as t11, "
                         "sum(other) as t12 from dm.shopping_behavior group by mobile,"
                         " month_no)t "%read_table).toPandas()
    else:
        data = pred_data

    if online:
        if not pred_data:
            raise TypeError('If this is used for on-line, the pred_data should not given!')

        data = online_data.toPandas()

    pred_data = np.array(data.iloc[:, 1:]).reshape(-1, 12, 12)
    mobile = pd.DataFrame(data['mobile'].unique(), columns=['mobile'])

    model = tf.keras.models.load_model(model_path + '/LSTM_regression.h5')
    # this is the all the needed prediction score
    pred = model.predict(pred_data)
    # if I have getted the mobile, just combine the mobile and prediction two columns
    out = np.concatenate((np.array(mobile).reshape(-1,1), pred), axis=1)
    out = pd.DataFrame(out, columns=['mobile', 'score'])

    res = spark.createDataFrame(out)#.withColumn('stat_date', lit(stat_date))

    # insert result to Hive table
    res.createOrReplaceTempView('score_re')
    # in case the table is not present, create a table by myself
    if(is_first_time):
        #spark.sql("drop table %s" % res_write_table)
        spark.sql("create table %s(mobile string, score double) partitioned by (stat_date string) stored as orc"%res_write_table)

    in_query = "insert overwrite table %s partition (stat_date=\'%s') select * from score_re"%(res_write_table, stat_date)
    print_sql(in_query)

    spark.sql(in_query)

    print('sum-score task is Finished!!!')

    # release the memory
    spark.catalog.dropTempView('score_re')


# Because of this .py is just used by myself, use the __name__ attribution
if __name__ == '__main__':
    import argparse
    import time

    parse = argparse.ArgumentParser()
    parse.add_argument('--local', type=bool, help='use local or yarn local model.')
    parse.add_argument('--stat_date', type=str, help='which partition to be used to save data.')
    parse.add_argument('--is_first_time', type=bool, help='whether this is first time to run code.')
    args = vars(parse.parse_args())

    is_local = args['local']
    stat_date = args['stat_date']

    spark = SparkSession.builder.appName('TF').enableHiveSupport().getOrCreate()
    enable_arrow(False)

    if is_local:
        model_path = '/home/luguangqiang/pyspark/recommendation/model_path'
    else:
        model_path = './py2env_pick.zip/py2env_pick/lib/python2.7/site-packages/taobao_libs/models'
    # if the stat_date parameter is not given, use now date to write data to partition
    if stat_date is None:
        now = datetime.datetime.now()
        stat_date = now.strftime('%Y-%m-%d')
    # if the code is first time to run, create table
    if args['is_first_time']:
        is_first_time = args['is_first_time']

    s_t = time.time()
    multi_label_classification_sum_regression(model_path, stat_date=stat_date)
    print('Total process use %.2f seconds'%(time.time() - s_t))

