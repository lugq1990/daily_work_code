# -*- coding:utf-8 -*-
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, window

if __name__ == '__main__':
    if len(sys.argv) != 4 and len(sys.argv) != 5:
        print('Not statisfied parameters')
        sys.exit(-1)

    host = sys.argv[1]
    port = int(sys.argv[2])
    windowSize = int(sys.argv[3])
    slideSize = int(sys.argv[4])
    if slideSize > windowSize:
        print('slide size must be smaller than the window size')
    slideDuration = '{} seconds'.format(slideSize)
    windowDuration = '{} seconds'.format(windowSize)

    spark = SparkSession.builder.appName('windowBased').getOrCreate()

    lines = spark.readStream.format('socket').option('host', host)\
        .option('port', port)\
        .option('includeTimestamp','true')\
        .load()

    words = lines.select(explode(split(lines.value, ' ')).alias('word'))
    # use the window based to compute the word count
    windowsCount = words.groupBy(window(words.timestamp, windowSize, slideSize), words.word)\
    .count().orderBy('window')

    query = windowsCount.writeSteam.outputMode('complete').format('console').option('truncate','false').start()

    query.awaitTermination()


