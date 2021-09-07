import requests
from bs4 import BeautifulSoup

url = 'https://i.51job.com/resume/standard_resume.php?lang=c&resumeid=303176251'
data = requests.get(url)

data = BeautifulSoup(data.content, 'html.parser')
