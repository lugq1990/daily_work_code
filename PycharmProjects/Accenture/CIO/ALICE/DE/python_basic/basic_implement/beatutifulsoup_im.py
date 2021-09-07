# -*- coding:utf-8 -*-
"""This is implement use case for web spider for insight website,
link is:https://inshorts.com/en/read"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup


seed_urls = ['https://inshorts.com/en/read/technology',
             'https://inshorts.com/en/read/sports',
             'https://inshorts.com/en/read/world']


def build_data(seed_urls):
    news_data = []
    for url in seed_urls:
        news_category = url.split('/')[-1]
        data = requests.get(url)
        soup = BeautifulSoup(data.content, 'html.parser')

        news = [{'news_headline': headline.find('span', attrs={"itemprop": "headline"}).string,
                 'news_article': article.find('div', attrs={'itemprop': "articleBody"}).string,
                 'news_category': news_category} for headline, article in zip(
            soup.find_all('div', class_='news-card-title news-right-box'),
            soup.find_all('div', class_='news-card-content news-right-box'))]

        news_data.extend(news)
    df = pd.DataFrame(news_data)
    df = df[['news_headline', 'news_article', 'news_category']]
    return df


df = build_data(seed_urls)
print(df.head())