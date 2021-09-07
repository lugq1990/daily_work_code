# -*- coding:utf-8 -*-
from elasticsearch import Elasticsearch

content = list()
sentiment = list()

es = Elasticsearch(hosts='192.168.1.34', db=0)
doc = {'size':10000, 'query':{'match_all':{}}}

index = 'nmas'
doc_type = 'public_opinion_result'

# for now, there is just about 13k datasets.
res1 = es.search(index=index, doc_type=doc_type, body=doc, scroll='1m')
scroll = res1['_scroll_id']
res2 = es.scroll(scroll_id=scroll, scroll='1m')

res1_data = res1['hits']['hits']
res2_data = res2['hits']['hits']

content_list = list()
sentiment_list = list()

for i in range(len(res1_data)):
    op_content = res1_data[i]['_source']['opinion_content']
    op_sen = res1_data[i]['_source']['opinion_sentiment']
    if op_content is None:
        content_list.append('Nan')
    if op_sen is None:
        sentiment_list.append('Nan')
    content_list.append(op_content)
    sentiment_list.append(op_sen)

for j in range(len(res2_data)):
    op_content = res1_data[j]['_source']['opinion_content']
    op_sen = res1_data[j]['_source']['opinion_sentiment']
    if op_content is None:
        content_list.append('Nan')
    if op_sen is None:
        sentiment_list.append('Nan')
    content_list.append(op_content)
    sentiment_list.append(op_sen)