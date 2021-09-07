# -*- coding:utf-8 -*-
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split
import sys

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Not satisfied parameters')
        sys.exit(-1)

    boostrapServers = sys.argv[1]
    subscribeType = sys.argv[2]
    topics = sys.argv[3]

    spark = SparkSession.builder.appName('structedKafka').getOrCreate()

    # get the dataset from the kafka topic as a dataset
    lines = spark.readStream.format('kafka')\
        .option('kafka.boostrap.serves',boostrapServers)\
        .option(subscribeType,topics)\
        .load.selectExpr('CAST(value AS String)')

    words = lines.select(explode(split(lines.value, ' ')).alias('word'))

    # using group by for word count
    wordCounts = words.groupBy('word').count()

    # start to run the word count and print the result to console
    query = wordCounts.writeStream.outputMode('complete').format('console').start()

    query.awaitTermination()
