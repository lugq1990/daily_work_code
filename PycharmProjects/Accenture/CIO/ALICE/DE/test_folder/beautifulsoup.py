# -*- coding:utf-8 -*-
import urllib
from bs4 import BeautifulSoup
import requests

url = 'http://forecast.weather.gov/MapClick.php?lat=37.7772&lon=-122.4168'
page = requests.get(url)
soup = BeautifulSoup(page, 'html.parser')