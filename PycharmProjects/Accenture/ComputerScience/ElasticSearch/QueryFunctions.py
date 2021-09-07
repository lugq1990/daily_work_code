# -*- coding:utf-8 -*-
"""This is just used for querying using es search engine, Just like some build-in query function."""
from elasticsearch import Elasticsearch
import json

es = Elasticsearch()

index_name = 'test'

def s(body):
    return es.search(index=index_name, doc_type=index_name, body=body)

# 1.First is to get all the data and sort return result by age desc
body_full = {
    'query':{'match_all':{}},
    'sort':[{'age': 'desc'}],
    '_source':['name', 'age', 'department'],
    'from':0,
    'size':10
}

# 2. Get same restriction for both element
body_same_res = {'query':{
    'bool':{
        'filter':[
            {'match_phrase': {'name':'lugq'}},
            {'match': {'age':20}}
        ]
    }
}}


# 3.Get some phrase same time
body_phrase = {'query':
                   {'bool':
                        {'must':
                             {'match':
                                  {'name':
                                       {'query':'lugq liu frank',
                                        'minimum_should_match':'20%'}}}}}}

# 4. Get some phrase based on and or
body_phrase_and = {'query':
                       {'match':
                            {'name':
                                 {'query':'lugq Compunication',
                                  'operator': 'or'}}}}


# 5.Based on name and age
body_name_age = {'query':
                     {'bool':
                          {'must':[
                               {'match':{'name': 'lugq'}},
                           {'range':{'age':{'gt':10}}}]
                          }}}

# 6.Using fuzziness
body_fuzzi = {'query':
                  {'match':
                       {'name':
                            {'query':'lugqgg',
                             'fuzziness': 'AUTO'}}}}


if __name__ == '__main__':
    # Get all result
    print(json.dumps(s(body_fuzzi)['hits'], sort_keys=True, indent=4, separators=(',', ':')))