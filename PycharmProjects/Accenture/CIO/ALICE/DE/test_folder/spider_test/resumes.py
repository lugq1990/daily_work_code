# -*- coding:utf-8 -*-
import requests
from bs4 import BeautifulSoup

url = 'https://www.runoob.com/regexp/regexp-syntax.html'
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

data = requests.get(url, headers=headers)

soup = BeautifulSoup(data.content, 'html.parser')

