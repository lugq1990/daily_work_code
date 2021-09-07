# -*- coding:utf-8 -*-
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Not statified parameters')
        sys.exit(-1)

    host = sys.argv[1]
    port = sys.argv[2]

    spark = SparkSession.builder.appName('StructuredHost').getOrCreate()

    # get the streaming data from the host, load() will get the data
    lines = spark.readStream.format('scoket').option('host', host).option('port',port).load()

    # split the line to be word
    words = lines.select(explode(split(lines.value,' ')).alias('word'))

    query = words.writeStream.outputMode('complete').format('console').start()

    query.awaitTermination()