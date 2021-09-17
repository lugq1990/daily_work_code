from elasticsearch import Elasticsearch, RequestsHttpConnection
from elasticsearch_dsl import Search
from requests_aws4auth import AWS4Auth

access_key = "AKIARLSQS4QEZIUSERGE"
secret_key = "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"
region = 'us-east-1'
awsauth = AWS4Auth(access_key, secret_key, region, 'es')

url = "https://vpc-aliceportal-30899-dev-kavpqvwplcis2ts4ii6jrs7hku.us-east-1.es.amazonaws.com"

es = Elasticsearch(
    hosts=[url],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

index = 'legal_smart_en'

s = Search(using=es, index=index)
out = s.execute()

total = out['hits']['total'].value
hits = out['hits']['hits']

print("There are {} total contracts, get {} hits".format(total, len(hits)))