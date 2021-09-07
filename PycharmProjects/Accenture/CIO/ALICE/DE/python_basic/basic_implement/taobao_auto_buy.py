# -*- coding:utf-8 -*-
"""
This is implement with selenium to buy things with python.

@author: Guangqiang.lu
"""
from selenium import webdriver

browser = webdriver.Chrome()

# go to main web
browser.get("https://www.taobao.com")

browser.find_element_by_link_text("登录").click()

browser.get("https://cart.taobao.com/cart.htm")




