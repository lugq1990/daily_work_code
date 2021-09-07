# -*- coding:utf-8 -*-
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType
import numpy as np
from pyspark import SparkContext


spark = SparkSession.builder.enableHiveSupport().getOrCreate()
sc = SparkContext.getOrCreate()

cols = "page_white_list_id,page_media_code,page_origin_url,page_meta_description,page_meta_keywords,page_title,page_source,page_publish_time,page_origin_source,page_content,page_media_type,page_domain_type String"

# schema = StringType(StructField(filename, StringType(), True) for filename in cols.split(','))


path = '/apps/hive/warehouse/artile'

# data = sc.textFile(path + '/tfidf.txt')
#
# data = data.toDF(cols)

data = spark.read.format('libsvm').option("delimiter", "\001").load(path+'/tfidf.txt')
data.write.parquet(path+'/new_result.parquet')
print('finished!')


