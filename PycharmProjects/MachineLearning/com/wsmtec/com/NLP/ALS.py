# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.ml.recommendation import ALS
import pymysql
import logging
from pyspark.ml.feature import StringIndexer, IndexToString
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


sc = SparkContext()
spark = SparkSession.builder.enableHiveSupport().getOrCreate()

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)

connection = pymysql.connect(user='zhanghui',password='zhanghui',database='model_data',host='10.1.36.18',charset='utf8')
#query = 'select mobile , label2, count(1) as rating from (select *  from model_data.tb_item_classified where label2 is not null and price !=0 and status != "已取消")t group by mobile, label2'
query2 = "select mobile, label1, count(1) from (select * from model_data.tb_item_classified where label2 is not null and price > 0 and status !='已取消')t group by mobile, label1"
re = pd.read_sql(query2, con=connection)
re.columns = ['userId_raw', 'itemId_raw', 'rating']


df_read = spark.createDataFrame(re)

# convert the string-type column to be inteter because of the ALS needed[userID , itemID] need both int-type
df_userid = StringIndexer(inputCol='userId_raw', outputCol='userId').fit(df_read).transform(df_read)
df = StringIndexer(inputCol='itemId_raw', outputCol='itemId').fit(df_userid).transform(df_userid)

# split the datasets to be train and test datasets
(training, test) = df.randomSplit([.8, .2])
# now start to construct the model
als = ALS(maxIter=3, regParam=1,rank=10, itemCol='itemId', ratingCol='rating', userCol='userId', coldStartStrategy='drop')
model = als.fit(training)
# get the prediction from the model
pred = model.transform(test)
# then evualuate the model
evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
print('model RMSE is %.5f'%(evaluator.evaluate(pred)))

# use the cross-validation to choose the best model parameters
# paramGrid = ParamGridBuilder().addGrid(als.maxIter,[5, 10, 15])\
#     .addGrid(als.regParam,[.001, .01, .1])\
#     .addGrid(als.rank,[5, 10, 15])\
#     .addGrid(als.coldStartStrategy, ['drop', 'nan'])\
#     .build()
#
# cv = CrossValidator(estimator=als, estimatorParamMaps=paramGrid, numFolds=5, evaluator=evaluator)
# cv_model = cv.fit(training)
# pred_cv = cv_model.transform(test)
# print('Using the cross-validation RMSE =%.6f'%(evaluator.evaluate(pred_cv)))


# make a function to evaluate different params
def recommend(max_iter=10, reg_param=.01, rank=5):
    als = ALS(maxIter=max_iter, regParam=reg_param, rank=rank,itemCol='itemId',
              ratingCol='rating', userCol='userId', coldStartStrategy='drop')
    model = als.fit(training)
    pred = model.transform(test)
    evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
    rmse = evaluator.evaluate(pred)
    print('model RMSE is %.5f' % rmse)
    return rmse, model


""" because of the spark StringIndexer for the recommended user data type is Int,
    I can not reconvert it to orginial
    so use the sklearn to encode the string columns"""
from sklearn.preprocessing import LabelEncoder
en1 = LabelEncoder().fit(re['userId_raw'])
en2 = LabelEncoder().fit(re['itemId_raw'])
le1 = pd.DataFrame(en1.transform(re['userId_raw']), columns=['userId'])
le2 = pd.DataFrame(en2.transform(re['itemId_raw']), columns=['itemId'])
# concat all the data and result
re_new = pd.concat((re, le1, le2), axis=1)

df = spark.createDataFrame(re_new)
(training, test) = df.randomSplit([.8, .2])
# start to train the model
model = als.fit(training)
pred = model.transform(test)
print('model RMSE is %.5f'%(evaluator.evaluate(pred)))

# use the trained model to recommend the result to all the users
userRecs = model.recommendForAllUsers(125)
# explode the result recommendation to be multi for each person
userRecs.createOrReplaceTempView('t1')
userRecs_ex = spark.sql('select userId,explode(recommendations) as recommendations from t1')\
    .select(['userId','recommendations.itemId','recommendations.rating']).toPandas()


# start to re-convert the result to the original label columns
org_user = pd.DataFrame(en1.inverse_transform(userRecs_ex['userId']), columns=['org_user'])
# org_item = np.empty((userRecs_ex.shape[0], 2))
# item_df = userRecs_ex['recommendations']
# for i in range(userRecs_ex.shape[0]):
#     org_item[i, 0] = item_df[i].itemId
#     org_item[i, 1] = item_df[i].rating
org_item_name = pd.DataFrame(en2.inverse_transform(userRecs_ex['itemId'].astype(np.int)), columns=['org_item'])
# org_item_rec = pd.DataFrame(org_item, columns=['itemId','recommendation'])

# combine all the columns to be one dataframe
out = pd.concat((userRecs_ex, org_user, org_item_name), axis=1)
result = spark.createDataFrame(out)
# use the spark sql function pivot to make the column value to be the row value
res_pivot = result.groupBy('org_user').pivot('org_item').sum('rating')
# Because of the gc overhead, just use gc to collect the un-used data
import gc
del(org_item_name)
del(org_user)
del(out)
del(userRecs_ex)
del(df_read)
gc.collect()

result_pandas = res_pivot.toPandas()
result_pandas.to_csv('new_result.csv', index=False)
# result_pandas.columns = ['mobile','phoneup','other','publisher','medician',' matierials',' travals','homeElec','tools','commodity','clothings','unnecces','foods']

# write the dataset to mysql
# result_pandas.to_sql(con=connection, name='recommendResult')