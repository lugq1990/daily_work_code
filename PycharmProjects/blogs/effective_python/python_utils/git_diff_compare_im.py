"""Git difference comparation implement"""

import os
import difflib

file_list = ['sample1.txt', 'sample2.txt']

res_list = []
for file in file_list:
    with open(file, 'r') as f:
        data = [x.replace('\n', '') for x in f.readlines()]
        res_list.append(data)
        
        
print("\n".join(difflib.unified_diff(*res_list)))