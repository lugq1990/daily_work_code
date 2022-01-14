"""Util functionality to get douban books top 500 basic information and rating info.

logic is to call requests with douban website and use bs4 to extract HTML data and make it into 
dict, what we get is some columns and data in one file.

column supported:
    ['作者', '出版社', '原作名', '出版年', '页数', '定价', '装帧', 'ISBN', '标签', 'name',
       'score', 'rating', 'per_list', 'intro', '5star', '4star', '3star',
       '2star', '1star', '出品方', '副标题', '丛书']
"""

import re
from bs4 import BeautifulSoup
import requests
import time
import os
import json
import warnings
import random
import re
import copy
import sys
import pandas as pd

warnings.simplefilter("ignore")

# add new functionality that will store the IP that is workable, if that IP is not workable
# then will try a new one, if that is workable, will replace current one.
global workable_proxy 
workable_proxy = ''

base_top_url = "https://www.douban.com/doulist/45298673/?start={}&sort=seq&playable=0&sub_type="


# this is to get full list of free IPs, so that we could get more better choices
def get_ip_ports():
    url = 'https://free-proxy-list.net/'
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Cafari/537.36'}
    source = str(requests.get(url, headers=headers, timeout=10).text)
    data = [list(filter(None, i))[0] for i in re.findall('<td class="hm">(.*?)</td>|<td>(.*?)</td>', source)]
    groupings = [dict(zip(['ip', 'port', 'code', 'using_anonymous'], data[i:i+4])) for i in range(0, len(data), 4)]

    print("Get {} IPs".format(len(groupings)))

    ip_ports = [x.get('ip') + ":"+ x.get('port') for x in groupings if x.get('ip').count('.') == 3]

    return ip_ports


# add with random headers
def get_headers():
    USER_AGENT_LIST = [
              "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
              "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
              "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
              "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
              "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
              "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
              "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
              "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
              "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
              "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
              "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
              "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
              "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
              "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
              "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
              "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52",
          ]
    agent = random.choice(USER_AGENT_LIST)

    headers = {
        "User-Agent":
            agent
        } 

    return headers


def get_proxy():
    ip_ports = get_ip_ports()
    tmp_ip_list = ip_ports[:40] 
    random_ip = random.choice(tmp_ip_list)
    proxy = {
        'https': random_ip,
        'http': random_ip
        }
    return proxy


def call_api(url):
    get_res = False
    while not get_res:
        try:
            # will make random IP and random headers.  
            # if workable_proxy is None:
            #     proxy = get_proxy()   
            # remove `proxy`, as even in notebook, it is slow!
            response = requests.get(url, headers=get_headers(), timeout=20)
            get_res = True
            
            return response
        except Exception as e:
            # print("When try to call API get warning: {}\n Now will give new try.".format(e))
            workable_proxy = None


# this is to get full book list function
def get_book_list():
    book_list = []

    for i in range(0, 1):
        response = call_api(base_top_url.format(25*i))
        bs = BeautifulSoup(response.content, 'html.parser')
        
        res = bs.find_all('div', class_='title')

        for j in range(len(res)):
            book_dict = {}
            book_dict['name'] = res[j].text.replace('\n', '').strip()
            book_dict['link'] = res[j].find('a').get('href')

            book_list.append(book_dict)
        
        # time.sleep(1)
        print("already get {} pages.".format(i))
        
    return book_list


# this is to get each book information with requests.

def get_book_info(book_info_dict):
    url = book_info_dict.get('link')
    book_name = book_info_dict.get('name')

    response = requests.get(url, headers=get_headers())

    bs = BeautifulSoup(response.content, 'html.parser')

    out = {}

    try:
        book_info_list = bs.find("div", id='info').text.split('\n')
        book_info_list = [x.strip() for x in book_info_list if x.strip() != '']
        single_book_basic_info = [book_info_list[0] +" "+ book_info_list[1]] + book_info_list[2:]
    except:
        print("book: {} has some problem to get info".format(book_name))

    try:
        tag_list = bs.find("div", id='db-tags-section').text.split("\n")
        tag_list = ','.join([x.replace('\xa0', '').strip() for x in tag_list if x.strip() != ''][2:5])
    except:
        print("book: {} don't have tags".format(book_name))
        tag_list = ''

    # make it as dict

    for x in single_book_basic_info:
        s = x.split(":")
        if len(s) < 2 or s[1] == '':
            continue
        
        out[s[0]] = s[1].strip()
    out['标签'] = tag_list

    time.sleep(1)

    return out

def get_ratings(book_info_dict):
    rating_wrong_book_list = []
    
    new_url = book_info_dict.get('link')
    book_name= book_info_dict.get('name')

    response = requests.get(new_url, headers=get_headers())

    bs = BeautifulSoup(response.content, 'html.parser')

    out = {'name':book_name}

    try:
        # get score
        try:
            score = bs.find("strong", class_="ll rating_num ").text.strip()
        except:
            try:
                score = bs.find("strong", class_="ll rating_num").text.strip()
            except:
                print("Couldn't get score for book:".format(book_name))

        # get rating number
        rating = bs.find('a', class_='rating_people').text

        # get each rating percentage
        per_list = ','.join([x.text for x in bs.find_all('span', class_='rating_per')])

        # get intro
        intro = bs.find('div', class_='intro').text

        out['score'] = score
        out['rating'] = rating
        out['per_list'] = per_list
        out['intro'] = intro
        
    except:
        print("book: {} has some problem to get info".format(book_info_dict.get('name')))
        rating_wrong_book_list.append(book_name)

    # this is to add with ratings
    try:
        p_list = out.get('per_list').split(',')
    except:
        print("Book:{} don't have rating.".format(out.get('name')))
    l_list = ['5star', '4star', '3star', '2star', '1star']
    for p, l in zip(*[p_list, l_list]):
        out[l] = p

    return out


if __name__ == '__main__':
    book_list = get_book_list()
    # book_list = [{'link': 'https://book.douban.com/subject/1013208/', 'name': '如何阅读一本书'},
    #  {'link': 'https://book.douban.com/subject/1775691/', 'name': '少有人走的路'},
    #  {'link': 'https://book.douban.com/subject/1012611/', 'name': '乌合之众'}]
    output_res = []
    output_res_rating = []

    if len(book_list) == 0:
        print("There is no books found!")
        sys.exit(1)

    for i in range(len(book_list)):
        print("Now is {}th book.".format(i))
        out = get_book_info(book_info_dict=book_list[i])
        # this is for getting basic infomation
        output_res.append(out)
        
        rating_out = get_ratings(book_info_dict=book_list[i])
        output_res_rating.append(rating_out)
        

    # what we need is just to merge these 2 lists
    assert len(output_res) == len(output_res_rating), "both basic and rating should be same length! but not"

    # copy a new `output_res` for final output
    final_out = copy.deepcopy(output_res)

    for i in range(len(final_out)):
        # as there are both dict, just update the first one will be fine
        final_out[i].update(output_res_rating[i])


    # last we just need to make it into a DataFrame for file use case
    df = pd.DataFrame(final_out)

    df.to_excel('books_info.xlsx', index=False)
