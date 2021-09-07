import requests
import re
import json
import pandas as pd
from bs4 import BeautifulSoup


headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'
}


def main(page_num):
    url_new = 'https://movie.douban.com/top250?start='+ str(page_num*25)+'&filter='

    res = request_dang(url_new)
    soup = BeautifulSoup(res, 'lxml')
    items = extract(soup)

    return items
        


def request_dang(url):
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            return res.text
    except requests.exceptions.RequestException:
        raise requests.exceptions.RequestException("Couldn't get url: {}".foramt(url))

def extract(soup):
    item_list = []
    result = soup.find(class_='grid_view').find_all('li')

    for item in result:
        item_name = item.find(class_='title').string
        item_img = item.find('a').find('img').get('src')
        item_index = item.find(class_='').string
        item_score = item.find(class_='rating_num').string
        item_author = item.find('p').text.strip().replace('\n', '')
        try:
            item_intr = item.find(class_='inq').string
        except:
            item_intr = ''
        item_number = item.find('span')

        print('爬取电影：' + item_index + ' | ' + item_name +' | ' + item_img +' | ' + item_score +' | ' + item_author +' | ' + item_intr )
    #    print('爬取电影：' + item_index + ' | ' + item_name  +' | ' + item_score  +' | ' + item_intr )
        item = item_name +' | ' + item_img +' | ' + item_score +' | ' + item_author +' | ' + item_intr 
        item_list.append(item)

    return item_list


if __name__ == '__main__':
    res = []

    for i in range(10):
        page_data = main(i)
        for item in page_data:
            res.append(item)

    # print("get data:", len(res))
    
    # write result
    with open("dangdang_book.txt", 'w', encoding='utf-8') as f:
        for item in res:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # df = pd.DataFrame(res)
    # df.to_csv("dangdang_books.csv", index=False)