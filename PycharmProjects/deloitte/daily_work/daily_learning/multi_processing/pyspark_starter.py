from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
    
sc = spark.sparkContext

sc.addPyFile("env.tar.gz")

data = [1, 2, 3, 4, 5]
distData = sc.parallelize(data)

print(distData.collect())