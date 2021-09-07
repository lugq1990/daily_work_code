# -*- coding:utf-8 -*-
from pyspark.sql import SparkSession
import pandas as pd
from pyspark.ml.feature import StringIndexer, IndexToString
from pyspark.ml.recommendation import ALS
from sklearn.preprocessing import LabelEncoder
import numpy as np
from pyspark.sql.functions import monotonically_increasing_id, udf
from pyspark.sql.types import StringType

model_path = '/home/luguangqiang/pyspark/recommendation/model_path'

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
from model_test.shopping_behavior_
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
        if stat_date is None:
            stat_date = '2018-03-28'
        else:
            if stat_date.__class__ == np.ndarray:
                try:
                    stat_date = stat_date.tolist()
                except TypeError:
                    raise 'Wrong type parameter give for stat_date, must be list, string or numpy.ndarray!'

            stat_date = ','.join(stat_date)

        data = spark.sql('select mobile as userId_raw, label as itemId_raw, cnt as rating  from dm.recmd_data where stat_date in ('+stat_date+')')
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

    spark.sql("""insert overwrite table model_test.recomend_result
                select * from recommend""")
    print('All finished!!!')

    # release the memory
    spark.catalog.dropTempView('recommend')


    # # inverse transform the result
    # conver_user = IndexToString(inputCol='userId', outputCol='userId_raw').transform(userRecs_ex)
    # conver_item = IndexToString(inputCol='itemId', outputCol='itemId_raw').transform(conver_user)
    #
    # # get the result dataFrame
    # res_df = conver_item.select(['userId_raw', 'itemId_raw', 'rating'])

    # returned object is a spark DataFrame

import tensorflow as tf

def multi_label_classification(online=False, online_data=None):

    if online:
        data = online_data.toPandas()
    else:
        data = spark.sql("""
select mobile,
t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12 from
(select mobile, month_no,
sum(recharge) as t1, sum(food) as t2, sum(commodity) as t3, sum(clothing) as t4, sum(tools) as t5, sum(elect) as t6,
sum(publish) as t7, sum(entert) as t8, sum(equip) as t9, sum(noness) as t10, sum(medicine) as t11, sum(other) as t12
from model_test.shopping_behavior group by mobile, month_no)t """).toPandas()

    # before we running the multi-score task, we first run sum-score problem
        score_pred_task(pred_data=data)
    print('The sum-score task is finished! Then run the multi-lable task.')

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
    res.createOrReplaceTempView('multi')

    # insert the result to Hive table
    spark.sql('insert overwrite table model_test.multi_label_res select * from multi')
    print('All finished')

    # all done, release the memory
    spark.catalog.dropTempView('multi')



# This function is used to predict the sum-score for each person, using LSTM model to predict
def score_pred_task(pred_data=None, online=False, online_data=None):
    # because the data for sum-score is same with multi-label, we can re-use it.
    if pred_data is None:
        data = spark.sql("""
        select mobile,
        t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12 from
        (select mobile, month_no,
        sum(recharge) as t1, sum(food) as t2, sum(commodity) as t3, sum(clothing) as t4, sum(tools) as t5, sum(elect) as t6,
        sum(publish) as t7, sum(entert) as t8, sum(equip) as t9, sum(noness) as t10, sum(medicine) as t11, sum(other) as t12
        from model_test.shopping_behavior group by mobile, month_no)t """).toPandas()
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

    res = spark.createDataFrame(out)

    # insert result to Hive table
    res.createOrReplaceTempView('score_re')
    spark.sql('insert into table model_test.user_regression_score select * from score_re')

    print('All Finished!!!')

    # release the memory
    spark.catalog.dropTempView('score_re')


spark = SparkSession.builder.enableHiveSupport().getOrCreate()

# als_recommend(stat_date=['2018-03-28', '2018-03-29'])
multi_label_classification()


