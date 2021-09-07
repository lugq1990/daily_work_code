# -*- coding:utf-8 -*-
"""This is to get the machine learning chinese words translation with requests,
as when I use ipython terminal, always with chinese character encoding error,
but here with pycharm to make the encoding with 'utf-8', then works
"""

import requests
from bs4 import BeautifulSoup
import os

path = "C:/Users/guangqiiang.lu/Documents/lugq/workings/201908/ml_words"

# have to set this to get the access to web when with 203 error
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
url = "https://www.jianshu.com/p/bf873155cb5d"

data = requests.get(url, headers=headers)

soup = BeautifulSoup(data.content, 'html.parser')

cl = "show-content-free"

whole_result = soup.find('div', class_=cl).find_all('p')

res_list = []
for f in whole_result:
    res_list.append(f.text.replace('\xa0', ''))

print('Get res:', res_list[:10])

# find the chinese characters
import re

pattern = r"[\u4e00-\u9fff]+"

words_dict = {}
# get word key and value
for word in res_list:
    word = word.strip()
    if len(re.findall(pattern, word)) == 0:
        # print('Get no chinese word:', word)
        continue
    s_index = word.index(re.findall(pattern, word)[0])
    k = word[:s_index].strip()
    v = word[s_index:].strip()
    words_dict[k] = v


# write the result to disk
import json
import codecs

# with ensure to make the chinese words ok!
with codecs.open(os.path.join(path, 'words_ml.txt'), 'w', encoding='utf-8') as f:
    json.dump(words_dict, f, ensure_ascii=False)


with open(os.path.join(path, 'words_ml.txt'), 'r', encoding='utf-8') as f:
    dic = f.readlines()[0]

import json

dic = json.loads(dic)


with open(os.path.join(path, 'words_ml_to_remember.txt'), 'w', encoding='utf-8') as f:
    for k, v in dic.items():
        f.write(k + ': ' + v + '\n')

