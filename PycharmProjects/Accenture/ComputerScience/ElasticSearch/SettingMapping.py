# -*- coding:utf-8 -*-
import os
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, streaming_bulk
from elasticsearch.exceptions import TransportError
import docx

path = "C:/Users/guangqiiang.lu/Documents/lugq/workings/201901"

def get_file_list():
    file_list = []

    # loop for this file dir name, if file endswith '.docx', here will read the docx to memory
    for doc in os.listdir(path):
        if doc.endswith('docx'):
            file_list.append(doc)
    return file_list

# Get docx file content to memory
def get_docx(f):
    doc = docx.Document(os.path.join(path, f))
    fulltext = []
    for para in doc.paragraphs:
        fulltext.append(para.text)

    return '\n'.join(fulltext)

file_list = get_file_list()
doc_data = [get_docx(f) for f in file_list]


# Here I just want to make one index if not exits, but also using the mapping and setting json.
def create_index(client, index):
    create_index_body = {
        'settings':{
            'number_of_shards':1,
            'number_of_replicas':0,

            # Custom analyser
            'analysis':{
                'analyzer':{
                    'trans':{
                        'type':'custom',
                        'tokenizer':'standard',
                        'filter':['lowercase']
                    }
                },

        'mappings':{
            'doc_bulk':{
                'properties':{
                    'title':{'type':'text'},
                    'context': {'type': 'text', 'analyzer': 'trans'}
                }
            }
        }
            }
        }
    }

    # Try to create the index, if exits, just pass, otherwise raise error
    try:
        client.indices.create(index=index, body=create_index_body)
    except TransportError as e:
        if e.error == 'index_already_exists_exception' or e.error == 'resource_already_exists_exception':
            pass
        else:
            raise

# Here I write one function to make the bulk json
# There is a problem that if I use this function, I can't do the right index.
def make_bulk_json(index, files, data, generator=False):
    if not generator:
        return [{'_index': index,
                    '_type': index,
                    '_id': j,
                    '_source':{'title': files[j],
                               'content': data[j]}}
                   for j in range(len(files))]
    else:
        for j in range(len(files)):
            yield {'_index': index,
                        '_type': index,
                        '_id': j,
                        '_source':{'title': files[j],
                                   'content': data[j]}}

# After make the json, then use the bulk or streamingbulk to do the index
def make_index(client, index, s_bulk=False):
    # actions = make_bulk_json(index, file_list, doc_data, generator=False)
    # Not use the function to build the index.
    actions = [{'_index': index,
                    '_type': index,
                    '_id': j,
                    '_source':{'title': file_list[j],
                               'content': doc_data[j]}}
                   for j in range(len(file_list))]

    if s_bulk:
        s, _ = streaming_bulk(client, actions)

    else:
        s, _ = bulk(client, actions)

    # After index process finished, then refresh the index
    client.indices.refresh(index=index)
    print('Index process success %s'%(str(s)))
    return 'All finished!'

def get_result(client, index):
    query_body = {
        'query':{
            'match_all':{}
        }
    }
    return client.search(index=index, doc_type=index, body=query_body)


if __name__ == '__main__':
    # Get the data from docx
    # doc_data = [get_docx(f) for f in file_list]

    es = Elasticsearch()
    index_name = 'doc_bulk'
    create_index(es, index_name)

    make_index(es, index_name, s_bulk=False)

    print(get_result(es, index_name)['hits']['hits'])


