# -*- coding:utf-8 -*-
from pyspark.sql import SparkSession
import sys
from pyspark.sql.functions import explode, split

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Not enough parameters!')
        sys.exit(-1)

    host = sys.argv[1]
    port = int(sys.argv[2])

    # constructe the spark
    spark = SparkSession.builder.getOrCreate()

    # read the streamming data from the spark
    lines = spark.readStream.format('socked').option('host', host).option('port', port).load()

    # start to run the query based on the received data
    words = lines.select(explode(split(lines.value, ' ')).alias('word')).groupBy('word').count()

    result = lines.writeStream.outputMode('complete').format('console').start()

    result.awaitTermination()

