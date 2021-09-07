# -*- coding:utf-8 -*-
"""This is to use github api to get some useful information for repository with requests"""


from requests import get
from re import search

git_url = 'https://github.com/lugq1990/LearningPapers.git'

url_expression = r'(https?://)?(www.)?github.com/[a-zA-Z0-9-]+/[a-zA-Z0-9-]+'
if not search(url_expression, git_url):
    raise ValueError("Not correct github url!")

repo_expression = r'/[a-zA-Z0-9-]+/[a-zA-Z0-9-]+'
# get the match result, get the match result
match = search(repo_expression, git_url)

repo_github_url = 'https://api.github.com/repos' + match.group()

print('Get repo url:', repo_github_url)

# get the response, make the result to json type
response = get(repo_github_url, headers={'Accept':'application/json'})

if response.status_code != 200:
    raise ConnectionError("Connection with error!")

resp_data = response.json()

expression_list = ['name', 'description', 'fork', 'size', 'language']
for e in expression_list:
    if e == 'size':
        print("%s: %.2f MB" % (e, resp_data[e] / 1024))
    else:
        print("%s: %s " % (e, resp_data[e]))
