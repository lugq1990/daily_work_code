# -*- coding:utf-8 -*-
from pyspark.sql import SparkSession
import argparse
from pyspark.sql.functions import monotonically_increasing_id


if __name__ == '__main__':
    path = '/home/tmp/lugq'
    parse = argparse.ArgumentParser()
    parse.add_argument('--n_start', type=int)
    parse.add_argument('--n_end', type=int)
    args = vars(parse.parse_args())

    start = args['n_start']
    end = args['n_end']

    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    #spark.conf.set("spark.sql.execution.arrow.enabled", "true")

    print('Show data!')
    spark.sql('select page_title,page_content from artile ').toPandas().to_parquet(path+'/respar.gzip')
    # df_new = df.withColumn('num', monotonically_increasing_id())
    # df.createOrReplaceTempView('t')

    # col_names = ["page_title","page_content","index"]
    # df = df.rdd.zipWithIndex().toDF(col_names)
    # print('start!!!')
    # print(df.agg({'index':'max'}).collect()[0])
    # print('how many:', df.count())
    # # df_new.filter(df_new.num > start).toPandas().to_parquet(path+'/respar'+str(end)+'.gzip')
    # print('Ending!')
    # print('max row:', df_new.agg({'num':'max'}).collect()[0])


    # sql = """select page_title,page_content from
    # (select row_number() over(order by page_white_list_id) as n, page_title,page_content from artile)t
    # where n < 10000
    # """
    # df_pandas = spark.sql(sql)
    # df_pandas.show()


    # df_pandas.toPandas().to_parquet(path+'/respar'+str(end)+'.gzip')
    # print('finished save data!')



