import requests
from bs4 import BeautifulSoup
import json
import re
from math import ceil
import random
import time
import pandas as pd


def get_city_list():
    """
    从链家获取所有收录的城市列表及对应网页的url
    :return: 城市名-url对(list)
    """

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11',
        'Accept': "*/*",
        'Cache-Control': "no-cache",
        'Accept-Encoding': "gzip, deflate",
        'Connection': "keep-alive",
        'cache-control': "no-cache"
    }
    url = "https://bj.fang.lianjia.com/loupan/pg"
    pattern = re.compile(r'href="//([a-z\.]+)" title=".*">(.+)</a>')

    try:
        response = requests.request("GET", url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
    except requests.ConnectionError as e:
        print('Error:', e.args)

    data = []
    cities = soup.find_all("div", {"class": "city-enum fl"})
    for char in cities:
        for city in char.find_all('a'):
            match = re.search(pattern, str(city))
            data.append({'url': match.group(1), 'name': match.group(2)})

    return data


def get_estate_data(city_list):
    """
    从链家爬取各城市房源信息（城市，楼盘名，单价）
    :param city_list: 城市名-url对(list)
    :return: 楼盘数据(pd.DataFrame)
    """

    user_agents = [
        'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11',
        'Opera/9.25 (Windows NT 5.1; U; en)',
        'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)',
        'Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.5 (like Gecko) (Kubuntu)',
        'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.8.0.12) Gecko/20070731 Ubuntu/dapper-security Firefox/1.5.0.12',
        'Lynx/2.8.5rel.1 libwww-FM/2.14 SSL-MM/1.4.1 GNUTLS/1.2.9',
        'Mozilla/5.0 (X11; Linux i686) AppleWebKit/535.7 (KHTML, like Gecko) Ubuntu/11.04 Chromium/16.0.912.77 Chrome/16.0.912.77 Safari/535.7',
        'Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:10.0) Gecko/20100101 Firefox/10.0',
    ]

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11',
        'Accept': "*/*",
        'Cache-Control': "no-cache",
        'Accept-Encoding': "gzip, deflate",
        'Connection': "keep-alive",
        'cache-control': "no-cache"
    }

    estate_data = []
    last_url = None    # 储存上一次访问的url，作为请求头的Referer来伪装爬虫
    for city in city_list:
        url = "https://" + city['url'] + "/loupan/pg"
        headers['User-Agent'] = random.choice(user_agents)    # 每个城市随机切换User-Agent
        try:
            response = requests.request("GET", url, headers=headers)
            print(f"开始爬取城市：{city['name']}，url：{url}")
            print(f"当前User-Agent: {headers['User-Agent']}")
            soup = BeautifulSoup(response.text, 'html.parser')
        except requests.ConnectionError as e:
            print('Error:', e.args)

        entry_div = soup.find("div", {"class": "resblock-have-find"})    # 定位到楼盘信息展示位置
        entries = re.search(r'<span class="value">(\d+)</span>', str(entry_div)).group(1)
        pages = ceil(int(entries) // 10 + 1)    # 计算要爬取的页数，链家每页展示10个楼盘
        print(f"{city['name']}共有楼盘{entries}个，预计爬取{pages}页")

        for i in range(1, pages + 1):
            page_url = url + str(i)    # 每一页的url在pg后面加上页码
            if last_url is not None:
                headers['Referer'] = last_url    # 上一次访问的url写入Referer
            last_url = page_url

            try:
                response = requests.request("GET", page_url, headers=headers)
                print(f"当前爬取{city['name']}-第{i}页/{pages}页")
                soup = BeautifulSoup(response.text, 'html.parser')
            except requests.ConnectionError as e:
                print('Error:', e.args)

            estate_div = soup.find_all("div", {"class": "resblock-desc-wrapper"})    # 定位到单个楼盘位置
            if not estate_div:
                print("当前页面已无楼盘信息")
                break
            for estate in estate_div:
                estate = str(estate)
                name = re.search(r'>(.+)</a>', estate).group(1)
                price = re.search(r'<span class="number">(.+)</span>', estate).group(1)
                estate_data.append([city['name'], name, price])

            t = random.uniform(-5, 5)
            time.sleep(10 + t)    # 以5-15秒间隔爬取，伪装爬虫

    result = pd.DataFrame(estate_data, columns=['city', 'name', 'price'])
    return result


if __name__ == "__main__":

    city_url = get_city_list()
    with open("../data/city_url.json", "w") as f:
        json.dump(city_url, f, ensure_ascii=False)

    with open("../data/city_url.json", "r") as f:
        city_list = json.load(f)
    data = get_estate_data(city_list)
    data.to_csv('../data/house_data.csv', index=False, encoding='utf-8-sig')
