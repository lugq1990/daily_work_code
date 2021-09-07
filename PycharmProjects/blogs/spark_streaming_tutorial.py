from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
from pyspark import SparkConf


# set port
conf = SparkConf().set("spark.ui.port",  "4041")


spark = SparkSession.builder.master('local[*]').config(conf=conf).getOrCreate()
sc = spark.sparkContext
ssc = StreamingContext(sc, 1)


# connect with socket
port = 8080
lines = ssc.socketTextStream("localhost", port)

# get words count
words = lines.flatMap(lambda x: x.split(' ')).map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)

# get output
words.pprint()


# start streaming context
ssc.start()
ssc.awaitTermination()
