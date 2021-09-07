# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""

import apache_beam as beam

with beam.Pipeline() as pipeline:
  p = (
    pipeline | "create " >> beam.Create([
    {'value': 1, 'name':'a'},
    {'value': 1, 'name':'b'},
    {'value': 2, 'name':'c'},
    {'value': 2, 'name':'d'},
  ]) | 'filter' >> beam.Filter(lambda x: x['value'] == 1)
    | 'print' >> beam.Map(print)
  )
