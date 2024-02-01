from memory_profiler import profile
from pyspark.sql import SparkSession


@profile
def my_func():
    spark = SparkSession.builder.config("spark.python.profile.memory", True).getOrCreate()
    
    df = spark.range(1000)
    return df.collect()


if __name__ == '__main__':
    my_func()