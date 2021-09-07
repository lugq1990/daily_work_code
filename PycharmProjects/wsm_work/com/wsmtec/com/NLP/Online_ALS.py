# -*- coding:utf-8 -*-
import numpy as np
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pyspark.sql.functions import udf, StringType, monotonically_increasing_id
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
import time
import argparse

sys.path.append("./py2env_pick.zip/py2env_pick/lib/python2.7/site-packages/")
read_table = 'dm.recmd_data'
write_table = 'dm.recomend_result'
is_first = False

def enable_arrow(init=True):
    if init:
        spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    else:
        pass

# In case there already exits spark, define a function to get singleton spark
def get_spark():
    if 'spark' not in globals():
        globals()['spark'] = SparkSession.builder.appName('Online_ALS').enableHiveSupport().getOrCreate()
    return globals()['spark']

# this is a function to make the result data to be input dataframe for ALS
def _conver_fn():
    # first convert the original dataframe using explode with the buying column
    org_data = spark.sql("""
select mobile, cast(times as double) from
(select mobile, explode(f) as times from
(select mobile, split(tmp,',') as f from
(select mobile, concat(t1,',',t2,',',t3,',',t4,',',t5,',',t6,',',t7,',',t8,',',t9,',',t10,',',t11,',',t12) as tmp from
(select mobile,
sum(elect) as t1,sum(food) as t2,sum(equip) as t3,sum(clothing) as t4,sum(recharge) as t5,sum(entert) as t6,
sum(commodity) as t7,sum(tools) as t8,sum(publish) as t9,sum(other) as t10,sum(noness) as t11,sum(medicine) as t12
from dm.shopping_behavior
group by mobile)ta)ta2)ta3 )Ta4
    """)

    # this is a private method for converting the index to item string, using spark udf
    def _con(x):
        if x % 12 == 0:
            return 'elect'
        elif x % 12 == 1:
            return 'food'
        elif x % 12 == 2:
            return 'equip'
        elif x % 12 == 3:
            return 'clothing'
        elif x % 12 == 4:
            return 'recharge'
        elif x % 12 == 5:
            return 'entert'
        elif x % 12 == 6:
            return 'commodity'
        elif x % 12 == 7:
            return 'tools'
        elif x % 12 == 8:
            return 'publish'
        elif x % 12 == 9:
            return 'other'
        elif x % 12 == 10:
            return 'noness'
        else:
            return 'medicine'

    udf_c = udf(_con, StringType())

    # then give the org_data with a monotony index, and use the index to make the item columns
    org_index_df = org_data.withColumn('index', monotonically_increasing_id())
    return org_index_df.withColumn('items', udf_c('index')).select('mobile','items','times').toDF('userId_raw', 'itemId_raw', 'rating')


def als_recommend(need_conver=False, on_line=False, online_data=None, stat_date=None):
    # if the data is converted, then just read it from Hive, otherwise convert it from the original data.
    # because original data is stored as partitioned, so have to stat_date parameter, if given, cast it to string
    if not need_conver:
        q_s = ''
        if stat_date is None:
            q_s = '\'' + '2018-03-28' + '\''
        else:
            if stat_date.__class__ == str:
                stat_date = stat_date.split(',')

            # loop the stat_date list, construct a sql statement
            if len(stat_date) > 1:
                for i in range(len(stat_date)):
                    if i == 0: q_s += '\''+ np.str(stat_date[i]) + '\''
                    else: q_s += ',' + '\'' + np.str(stat_date[i]) + '\''
            else: q_s = '\'' + stat_date[0] + '\''

        query_sql = "select mobile as userId_raw, label as itemId_raw, cnt as rating from %s where stat_date in (%s)"%(read_table,q_s)
        print('Executing SQL:',query_sql)

        data = spark.sql(query_sql)
    else:
        data = _conver_fn()

    if on_line:
        # if it's used for streaming, we have to get the new coming data from streaming program
        # and new data is must be same structure like basic data
        # in case the income DataFrame columns not like the basic, change the columns name
        data = data.unionAll(online_data.toDF('userId_raw', 'itemId_raw', 'rating'))

    # because spark IndexToString is stateless, can not convert the result to original
    # so use sklearn to process the data
    data_pd = data.toPandas()


    le_user = LabelEncoder().fit(data_pd['userId_raw'])
    le_items = LabelEncoder().fit(data_pd['itemId_raw'])

    user = le_user.transform(data_pd['userId_raw']).reshape(-1, 1)
    items = le_items.transform(data_pd['itemId_raw']).reshape(-1, 1)

    user = pd.DataFrame(user, columns=['userId_raw'])
    items = pd.DataFrame(items, columns=['itemId_raw'])
    rating = pd.DataFrame(data_pd['rating'])
    df_read = pd.concat((user, items, rating), axis=1)
    df_read.columns = ['userId', 'itemId', 'rating']

    df_read = spark.createDataFrame(df_read)


    # Because the original data is just String label for each class,
    # Here I use the StringIndexer to index it
    # Because I have to recommend for all the new data with the original data,
    # so the StringIndexer have to be trained
    #     string_user = StringIndexer(inputCol='userId_raw', outputCol='userId').fit(df_read)
    #     df_userid = string_user.transform(df_read)
    #     string_item = StringIndexer(inputCol='itemId_raw', outputCol='itemId').fit(df_userid)
    #     df = string_item.transform(df_userid)

    als = ALS(maxIter=3, regParam=1, rank=10, itemCol='itemId', ratingCol='rating', userCol='userId',
              coldStartStrategy='drop', nonnegative=True)
    model = als.fit(df_read)

    userRecs = model.recommendForAllUsers(12)
    userRecs.createOrReplaceTempView('t1')
    userRecs_ex = spark.sql('select userId, explode(recommendations) as recommendations from t1')\
        .select(['userId', 'recommendations.itemId', 'recommendations.rating'])

    userRecs_ex = userRecs_ex.toPandas()

    # conver the result numerical data to string
    user_org = pd.DataFrame(le_user.inverse_transform(userRecs_ex['userId']), columns=['userId'])
    item_org = pd.DataFrame(le_items.inverse_transform(userRecs_ex['itemId']), columns=['itemId'])
    rating_org = pd.DataFrame(userRecs_ex['rating'], columns=['rating'])
    res_pd = pd.concat((user_org, item_org, rating_org), axis=1)

    res_df = spark.createDataFrame(res_pd)

    # because of I just want to make items name as columns and rating as the value of the column, use pivot
    # res = res_df.groupBy('userId').pivot('itemId').sum('rating')


    res_df.createOrReplaceTempView('recommend')
    # spark.sql("""insert overwrite table model_test.recomend_result
    #     select userId, clothing,commodity,elect,entert,equip,food,medicine,noness,other,publish,recharge,tools from recommend""")

    # if table not exits, create table by code

    if(is_first):
        spark.sql("drop table %s"%write_table)
        spark.sql("create table %s(user_id string, item_id string, rating double) stored as orc" % write_table)

    in_sql = "insert overwrite table %s select * from recommend"%write_table
    print('Now Executing SQL: ', in_sql)

    spark.sql(in_sql)
    print('All finished!!!')

    # release the memory
    spark.catalog.dropTempView('recommend')


# If this module is imported by other .py, then this method is not going to work.
if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument('-s', '--stat_date', help='which partition to process', type=str)
    paser.add_argument('-w', '--read_table', help='which table will read from .', type=str)
    paser.add_argument('-r', '--write_table', help='which table to write data into.', type=str)
    paser.add_argument('-i', '--is_first_time', help='whether is the first time to run code.', type=bool)
    args = vars(paser.parse_args())

    stat_date = args['stat_date']

    if args['read_table'] is not None:
        read_table = args['read_table']
    if args['write_table'] is not None:
        write_table = args['write_table']
    if args['is_first_time']:
        is_first = args['is_first_time']

    s_t = time.time()
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    enable_arrow(False)

    if stat_date is None:
        als_recommend()
    else:
        als_recommend(stat_date=stat_date)

    print('All finished use %s seconds'%(time.time() - s_t))
