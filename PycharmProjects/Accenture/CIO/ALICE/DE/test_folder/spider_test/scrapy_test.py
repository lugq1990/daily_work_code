# -*- coding:utf-8 -*-
import scrapy


class QuoteSpider(scrapy.Spider):
    name = 'quotes'
    start_urls = ['http://quotes.toscrape.com/tag/books/']

    def parse(self, response):
        for quote in response.css('div.quote'):
            yield {'text': quote.css('span.text::text').get(),
                   'author': quote.xpath('span/small/text()').get(),
                   }
        next_page = response.css('li.next a::attr("href")').get()
        if next_page is not None:
            yield response.follow(next_page, self.parse)




"""to get the unput files in HDFS"""
from hdfs.ext.kerberos import KerberosClient
import os
import tempfile

client = KerberosClient("http://name-node.cioprd.local:50070;http://name-node2.cioprd.local:50070")

upload_hdfs_path = '/data/insight/cio/alice/contracts_files/catchupfiles_others'
local_path = '/mrsprd_data/Users/ngap.app.alice/tmp_files'

file_name = [x for x in os.listdir(local_path) if x.endswith('.txt')][0]
with open(os.path.join(local_path, file_name), 'r') as f:
    data = [x.replace('\n', '') for x in f.readlines()]

data = [x + '.txt' for x in data]

tmp_path = tempfile.mkdtemp()

already_download = client.list(upload_hdfs_path)
d = set(already_download)

d_new = [x.split('/')[-1] for x in data]
not_list = list(set(already_download) - set(d_new))

for f in not_list:
    client.delete(os.path.join(upload_hdfs_path, f))
d_new = [x.split('/')[-1] for x in data]
d_new = list(set(d_new) - set(already_download))
data_new = []
for f in data:
    if f.split('/')[-1] not in d_new:
        data_new.append(f)

data = data_new

sati = False
while not sati:
    down_list = []
    up_list = []
    try:
        for f in data[::-1]:
            name = f.split('/')[-1]
            already_download.append(f)
            down_list.append(client.download(local_path=os.path.join(tmp_path, name), hdfs_path=f, overwrite=True))
            up_list.append(client.upload(hdfs_path=os.path.join(upload_hdfs_path, name), local_path=os.path.join(tmp_path, name),overwrite=True))
            if data[::-1].index(f) % 1000 == 0:
                print("already upload %d files" % data.index(f))
        sati = True
    except:
        print('retry')

import time
def how_many():
    print("already download %d files" % len(client.list(upload_hdfs_path)))

def num_min():
    s = len(client.list(upload_hdfs_path))
    time.sleep(60)
    e = len(client.list(upload_hdfs_path))
    print("1 min with %d files" % (e - s))

