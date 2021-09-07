# Spark version
# import pyspark
# from pyspark.sql import SparkSession


# spark = SparkSession.builder.getOrCreate()

# sc = spark.sparkContext

# res = sc.parallelize([1, 2, 3]).map(lambda x: x * 2).reduce(lambda x, y: x +y)

# print(res)


# spark collect

# values = sc.parallelize([1, 2, 3, 4])
# total = values.reduce(lambda x, y: x+ y)
# scaled_value = values.map(lambda x: x / total)
# print(scaled_value.collect())


# beam version
import apache_beam as beam

# with beam.Pipeline() as pipe:
#     res = (
#         pipe
#         | 'create data' >> beam.Create([1, 2, 3])
#         | 'multiply' >> beam.Map(lambda x: x * 2)
#         | 'reduce' >> beam.CombineGlobally(sum)
#         | 'print' >> beam.Map(print)
#     )

# with beam.Pipeline() as pipe:
#     values = pipe | beam.Create([1,2, 3, 4])
#     total = values | beam.CombineGlobally(sum)
    
#     scaled_value = values | beam.Map(lambda x, total: x/total,  total=beam.pvalue.AsSingleton(total))
    
#     scaled_value | beam.Map(print)

import re

def count(words):
    (word, ones) = words
    return (word, sum(ones))

@beam.ptransform_fn
def CountWord(pcoll):
    return (
        pcoll
        # | "extract" >> beam.FlatMap(lambda x: re.findall(r'[A-Za-z\']', x))
        | "get" >> beam.FlatMap(lambda x: x.split(" "))
        | "map" >> beam.Map(lambda x: (x, 1))
        # | "count" >> beam.combiners.Count.PerElement()
        | "reduce" >> beam.GroupByKey()
        | "count" >> beam.Map(count)
    )

with beam.Pipeline() as pipe:
    res = (
        pipe
        | "create data" >> beam.Create(["This is 1", "This is 2"])
        # | "split" >> beam.Map(lambda x: x.split(' '))
        | "ptrans" >> CountWord()
        | 'encode' >> beam.Map(lambda x: x[0]).with_output_types(str)
        | "print" >> beam.Map(print)
    )