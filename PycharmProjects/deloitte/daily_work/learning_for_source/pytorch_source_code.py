import requests
from multiprocessing import Pool

headers = {'User-Agent': 'Mozilla/5.0'}



book_list = [{'name': '这些人，那些事', 'link': 'https://book.douban.com/subject/6388661/'},
 {'name': '牧羊少年奇幻之旅', 'link': 'https://book.douban.com/subject/3608208/'},
 {'name': '浪潮之巅', 'link': 'https://book.douban.com/subject/6709783/'},
 {'name': '挪威的森林', 'link': 'https://book.douban.com/subject/1046265/'},
 {'name': '菊与刀(插图评注版）', 'link': 'https://book.douban.com/subject/2340309/'},
 {'name': '一九八四', 'link': 'https://book.douban.com/subject/1858576/'},
 {'name': '史记', 'link': 'https://book.douban.com/subject/1836555/'},
 {'name': '查令十字街84号', 'link': 'https://book.douban.com/subject/1316648/'},
 {'name': '从一到无穷大', 'link': 'https://book.douban.com/subject/1102715/'},
 {'name': '爱的艺术', 'link': 'https://book.douban.com/subject/3026879/'},
 {'name': '决策与判断', 'link': 'https://book.douban.com/subject/1193621/'},
 {'name': '黑客与画家', 'link': 'https://book.douban.com/subject/6021440/'},
 {'name': '学会提问', 'link': 'https://book.douban.com/subject/1504957/'},
 {'name': '心是孤独的猎手', 'link': 'https://book.douban.com/subject/1424741/'},
 {'name': '点石成金', 'link': 'https://book.douban.com/subject/1827702/'},
 {'name': '自私的基因', 'link': 'https://book.douban.com/subject/1292405/'},
 {'name': '孤独六讲', 'link': 'https://book.douban.com/subject/4124727/'},
 {'name': '最好的告别', 'link': 'https://book.douban.com/subject/26576861/'},
 {'name': '卡拉马佐夫兄弟', 'link': 'https://book.douban.com/subject/1856494/'},
 {'name': '窗边的小豆豆', 'link': 'https://book.douban.com/subject/1007914/'},
 {'name': '刀锋', 'link': 'https://book.douban.com/subject/2035162/'},
 {'name': '三体Ⅱ', 'link': 'https://book.douban.com/subject/3066477/'},
 {'name': '故事', 'link': 'https://book.douban.com/subject/1115748/'},
 {'name': '亲爱的安德烈', 'link': 'https://book.douban.com/subject/3369793/'},
 {'name': '三体Ⅲ', 'link': 'https://book.douban.com/subject/5363767/'}]


def call_api(d):
        link = d.get('link')
        response = requests.get(link, headers=headers)
    
        print(response.status_code)

if __name__ == '__main__':
    
    with Pool(4) as p:
        p.map(call_api, book_list[:5])