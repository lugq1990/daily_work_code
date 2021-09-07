# -*- coding:utf-8 -*-

"""
    This is for the on-line model, abstract all the features to one function:
    One: For using the ALS to train the recommendation problem
    Two: For using LSTM to get the multi-label problem, I can just load the model
    Three: For using LSTM to get the sum-score for each person problem
"""
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
import numpy as np
import pandas as pd
import logging
import pymysql
import jieba.analyse as ana
import jieba.posseg as pseg
import time
from gensim.models import Word2Vec
from sklearn.externals import joblib
import keras

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

# This is the global variables
# model_path = 'F:\workingData\\201806\\recommendation\MultiLabel\multi_model'
model_path = '/home/lugq/pyspark/StructuredStreaming'
topk = 6
spark = SparkSession.builder.appName('Structured_Streaming').getOrCreate()

"""
    Bellow function:
    1.cut the Taobao item with attribution, and get the key word for each item information
    2.get the word2vec result
    3.get the TF-IDF weights for each word
    4.get the LSTM model for predicting for new coming item belongs to
    5.construct the data structure for ALS data input
    6.make ALS model for recommendation
    7.construct the recommendated data for the multi-label data
    8.construct the recommendated data for the only one result for Regression
    9.get the one result Regression for output
    10.get the multi-label classification for output
"""

def tmp_data():
    connection = pymysql.connect(user='zhanghui', password='zhanghui', database='model_data', host='10.1.36.18',
                                 charset='utf8')
    # query = 'select item_name,label as label from model_data.tb_item_label_cfm2 where label is not null and sorttime <= \'2018-03-01\' '
    query = 'select * from model_data.train_data_import_db'
    re = pd.read_sql(query, con=connection)

    from sklearn.preprocessing import LabelEncoder
    le_model = LabelEncoder().fit(re['label'])
    # save the label-encoder model to disk
    joblib.dump(le_model, model_path + '/label_encoder.h5')
    le_model = joblib.load(model_path + '/label_encoder.h5')
    new_label = le_model.transform(re['label']).astype(np.int32)
    items = np.array(re['item_name'])
    label = np.array(new_label)
    # get the random index for training, because that too much data, the model training time is really long.
    p = 1
    rand_choice = np.random.choice(a=[True, False], size=(items.shape[0]), p=(p, 1 - p))
    print('the sum of all sati data is ', np.sum(rand_choice))
    item_choice = items[rand_choice]
    label_choice = label[rand_choice]
    return item_choice, label_choice, le_model

def tmp_data_new_come():
    connection = pymysql.connect(user='zhanghui', password='zhanghui', database='model_data', host='10.1.36.18',
                                 charset='utf8')
    query = 'select mobile,substring(create_time,1,7) as month, item_name from model_data.tb_item_classified where label2 is not null and price > 0 and status !=\'已取消\' limit 1000'
    re = pd.read_sql(query, con=connection)
    mobile = np.asarray(re.ix[:,'mobile'])
    items = np.asarray(re.ix[:, 'item_name'])
    months = np.asarray(re.ix[:, 'month'])
    return mobile, items, months
# mobile, item_choice, months= tmp_data_new_come()

# this function is used to get data from Hive
def hive_data():
    data = spark.sql('select mobile, item_name from etl.bds_madrid_tb_orderdetails_d')
    data_df = data.toPandas()
    mobile = np.asarray(data_df['mobile'])
    items = np.asarray(data_df['item_name'])
    return mobile, items

mobile, item_choice = hive_data()


# This function is just to cut the each item sentence with returned wordcut list
# the 'data' param is just the receiving data, for example the streaming data
def cut_words_with_attr(data=item_choice, pos_allow=True, allow_pos=['n','nr']):
    return_sati_list = list()
    for i, d in enumerate(data):
        cut_data = pseg.cut(d)
        sati_sati = list()
        for word, flag in cut_data:
            if pos_allow:
                if(flag not in allow_pos):
                    continue
            sati_sati.append(word)
        return_sati_list.append(sati_sati)
        if(i % 20000 == 0 ):
            print('Now is step %d'%i)
    return return_sati_list
cut_words = cut_words_with_attr()

# This function is to get the each sentence's key-word
def get_keywords(data = item_choice, topk=4, iter=2):
    start_time = time.time()
    key_words = data
    key_words_return = list()
    for j in range(iter):
        print('Now in the iteration %d'%j)
        for i,d in enumerate(key_words):
            key_words_return.append(ana.extract_tags(d,topK=topk,allowPOS=('n','nr')))
            if(i%20000 == 0):
                print('now is the %d samples for iteration %d'%(i, j))
        key_words = list()
        for m in range(len(key_words_return)):
            key_words.append("".join(key_words_return[m]))
        if(j != iter - 1):
            key_words_return = list()
    print('The all process used %f seconds'%(time.time()-start_time))
    return key_words_return

    # GET all the non-null columns
    key_words_list = np.array(key_words)
    # get the null row nums
    null_col = np.empty_like(key_words_list)
    for i in range(len(key_words)):
        if (len(key_words[i]) == 0):
            null_col[i] = False
        else:
            null_col[i] = True
    # make the data type to be boolean
    null_col = null_col.astype(np.bool)
    # get the non-null cols
    key_words_training = key_words_list[null_col]
    return key_words_training

key_words_training = get_keywords(topk=topk, iter=1)


# This function to get the each key-word with the vecter directory
def get_word2vec():
    model = Word2Vec.load(model_path+'/word2vec.bin')
    wordsvec = model[model.wv.vocab]
    uni_words = list(model.wv.vocab)
    # make the key-words and word2vec result directory for bellow using the key-value
    res_dic = dict()
    for j in range(len(uni_words)):
        res_dic.update({uni_words[j]: wordsvec[j, :]})
    return res_dic

res_dic = get_word2vec()


# This function is used to load the TF-IDF weights for each word
# input param data is just coming data key-word list
def get_tfidf_weight(data = key_words_training):
    # IDF dictory
    r_idf = joblib.load(model_path + '/tfidf_dic.h5')
    # I have to compute the TF*IDF for the dataset
    conver_array = np.empty_like(data)
    for i in range(len(data)):
        tfidf_list = list()
        for j in range(len(data[i])):
            # first compute the tf value
            tf = data[i].count(data[i][j]) / len(data[i])
            if (data[i][j] not in r_idf.keys()):
                # idf = 1.   # change the not exist key-word for 0.0 means that the key-word shows in all corpus log2(n/n) =0.0
                idf = 0.0
            else:
                idf = r_idf[data[i][j]]
            tf_idf = tf * idf
            tfidf_list.extend([tf_idf])
        conver_array[i] = tfidf_list
    # This is the tf-idf weight result
    return conver_array
conver_array = get_tfidf_weight()


# This function is used to make the nlp classes classification model input data, with no label for on-line
def get_nlp_classification_data(data = np.array(key_words_training), conver_array = conver_array, res_dic = res_dic):
    # result dimension like num*100*5 tensor
    result_image = np.zeros((data.shape[0], topk, 100))
    # convert unique list to set is much faster
    res_keys = set(list(res_dic.keys()))
    # loop the all visuable vectors
    for i in range(data.shape[0]):
        for j in range(len(data[i])):
            # because of the now word2vec model is based on the all satisified word, need to get the satisified key-word vector
            if (data[i][j] not in res_keys):
                continue
            # add the tf-idf vector to the res
            result_image[i, j, :] = res_dic[data[i][j]] * conver_array[i][j]
    return result_image
result_image = get_nlp_classification_data()

# This function is used to get the Res-CNN for the classification result
def get_nlp_classification_result(data = result_image,need_inverse_trans=False):
    model = keras.models.load_model(model_path + '/item_classification.h5')
    pred = model.predict(data)
    res = np.argmax(pred, axis=1)
    if need_inverse_trans:
        le_model = joblib.load(model_path + '/label_encoder.h5')
        res = le_model.inverse_transform(res)
    return res

res = get_nlp_classification_result()
# becuase of the ALS model is builded on 12 classes, so here is just a function to convert the 125 classes to 12 classes
def conv_f(x):
    if x< 12:
        return 0
    elif x>=12 and x<24:
        return 1
    elif x>=24 and x<36:
        return 2
    elif x>=36 and x<48:
        return 3
    elif x>=48 and x<60:
        return 4
    elif x>=60 and x<72:
        return 5
    elif x>=72 and x<84:
        return 6
    elif x>=84 and x<90:
        return 7
    elif x>=90 and x<95:
        return 8
    elif x>=95 and x<100:
        return 9
    elif x>=100 and x<110:
        return 10
    else:
        return 11

tmp_d = np.array([conv_f(x) for x in res])
data_for_als = pd.DataFrame(np.concatenate((mobile.reshape(-1, 1), tmp_d.reshape(-1, 1)), axis=1), columns=['mobile', 'label'])


# because I want to make the data for the ALS model, so I just make the result data to spark DataFrame,
# and register a table and group by the model to get the final result
spark.createDataFrame(data_for_als).createOrReplaceTempView('t1')
als_df = spark.sql('select mobile , label , count(1) as rating from t1 group by mobile, label')



# This function is to recommendate the users' implicit features
def als_recommend(data=None):
    conf = SparkConf()
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    # if the data is not given, before I just read Mysql database
    if data is None:
        connection = pymysql.connect(user='zhanghui', password='zhanghui', database='model_data', host='10.1.36.18',
                                     charset='utf8')
        # query = 'select mobile , label2, count(1) as rating from (select *  from model_data.tb_item_classified where label2 is not null and price !=0 and status != "已取消")t group by mobile, label2'
        query2 = "select mobile, label1, count(1) from (select * from model_data.tb_item_classified where label2 is not null and price > 0 and status !='已取消')t group by mobile, label1"
        re = pd.read_sql(query2, con=connection)
        re.columns = ['userId_raw', 'itemId_raw', 'rating']
        df_read = spark.createDataFrame(re)
    else:
        # read the Hive warehouse, for now test, just assume that the data is the spark DataFrame

        df_read = data.selectExpr('mobile as userId_raw','label as itemId_raw', 'rating as rating')

    # Because the original data is just String label for each class,
    # Here I use the StringIndexer to index it
    # Because I have to recommend for all the new data with the original data,
    # so the StringIndexer have to be trained
    df_userid = StringIndexer(inputCol='userId_raw', outputCol='userId').fit(df_read).transform(df_read)
    df = StringIndexer(inputCol='itemId_raw', outputCol='itemId').fit(df_userid).transform(df_userid)

    als = ALS(maxIter=3, regParam=1, rank=10, itemCol='itemId', ratingCol='rating', userCol='userId',
              coldStartStrategy='drop')
    model = als.fit(df)
    userRecs = model.recommendForAllUsers(12)
    userRecs.createOrReplaceTempView('t1')
    userRecs_ex = spark.sql('select userId,explode(recommendations) as recommendations from t1') \
        .select(['userId', 'recommendations.itemId', 'recommendations.rating'])
    # returned object is a spark DataFrame
    return userRecs_ex
als_pred = als_recommend(data=als_df)
print('This is the end of the function:')
als_pred.show(truncate=False)



# This function is used to prediction the new come person for multi-label implement,
# For whether or not the indexed score columns appeared
import tensorflow as tf

def multi_label_classification(data=None):
    if data is None:
        new_path = 'F:\workingData\\201806\\recommendation\MultiLabel'
        df = pd.read_csv(new_path + '/train_data_12.csv')
        df.dropna(inplace=True)
        data = df.iloc[:, :145].drop('mobile', axis=1)
        label = df.iloc[:, 145:]
        # So the data must be size 144
        data = np.asarray(data).reshape(-1, 12, 12)
    else:
        pass
    # load the already trained model
    model = tf.keras.models.load_model(model_path + '/LSTM_multi_class.h5')
    pred = model.predict(data)
    f = lambda x:1 if x>=.5 else 0
    pred = pd.DataFrame(pred).applymap(f)
    # return pred is DataFrame that means for whether or not the indexed score column appeared
    return pred


# This function is used to predict the sum-score for each person, using LSTM model to predict
def score_pred(data = None):
    if data is None:
        path = 'F:\workingData\\201806\ALS\\train_data'
        df = pd.read_csv(path + '/train_data_cnn_11.csv')

        # get the data and label column data
        mobile = df.iloc[:, 0]
        data = df.iloc[:, 1:-1]
        label = df.iloc[:, -1]
        # data = np.array(data).reshape(-1, 53, 12)
        # This data is batch_size*12*12
        data = np.array(data).reshape(-1, 12, 12)
        label = np.array(label).reshape(-1, 1)
    else:
        pass
    model = tf.keras.models.load_model(model_path + '/LSTM_regression.h5')
    # this is the all the needed prediction score
    pred = model.predict(data)
    # if I have getted the mobile, just combine the mobile and prediction two columns
    out = np.concatenate((np.array(mobile).reshape(-1,1), pred), axis=1)
    return out



