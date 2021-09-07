# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""
import os

url = "https://github.com/lugq1990/LearningPapers/tree/master/Books"
local_path = 'C:/Users/guangqiiang.lu/Documents/lugq/MachineLearningDoc/LearningPapers/Books'
file_output_path = 'C:/Users/guangqiiang.lu/python_code/python_data_analysis'

output_str = ''

folder_list = os.listdir(local_path)

for folder in folder_list:
    output_str += "## " + folder + "\n"
    for file in os.listdir(os.path.join(local_path, folder)):
        url_file_name = file.replace(' ', '%20')
        file_str = "  * [{}]({})".format(file.replace('.pdf', ''), '/'.join([url, folder, url_file_name]))
        output_str += file_str + '\n'
    output_str += '\n\n'

with open(os.path.join(file_output_path, 'github_readme.txt'), 'w', encoding='utf-8') as f:
    f.write(output_str)
