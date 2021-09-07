from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split


spark = SparkSession.builder.appName("test_structured_streaming").getOrCreate()


lines = spark.readStream.format("socket").option("host", 'localhost').option('port', 8080).load()

words = lines.select(explode(split(lines.value, ' ')).alias("word"))

word_counts = words.groupBy("word").count()

query = word_counts.writeStream.outputMode("complete").format("console").start()

query.awaitTermination()
