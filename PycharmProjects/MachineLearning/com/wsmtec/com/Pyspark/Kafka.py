# -*- coding:utf-8 -*-
import sys

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Not statisfied params')
        exit(-1)

    sc = SparkContext(appName='JustTest')
    batchInterval = 10
    ssc = StreamingContext(sc, batchDuration=batchInterval)

    lines = ssc.socketTextStream(sys.argv[1], int(sys.argv[2]))
    print('This is the getted Data:')
    lines.pprint()
    counts = lines.flatMap(lambda line: line.split(' '))\
        .map(lambda word:(word, 1))\
        .reduceByKey(lambda a,b:a+b)
    print('This is the Result:')
    counts.pprint()

    # start the streaming context
    ssc.start()
    # difine the await time
    ssc.awaitTermination(batchInterval *2)
