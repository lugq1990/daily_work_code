# -*- coding:utf-8 -*-
import os

path = 'C:/Users/guangqiiang.lu/Documents/lugq/workings/201905/file_compare/'
file1 = '0F0Y0VB313G.txt'
file2 = '0F0Y0VB313G (1).txt'

out1 = []
with open(os.path.join(path, file1), 'r') as f1:
    lines = f1.readlines()
    if len(lines) != 0:
        out1.extend(lines)
out1 = out1[0][0]
out1.replace('//t', '')
out1.replace('//m', '')
out1.replace('//', '')

out = []
with open(os.path.join(path, file2), 'r') as f2:
    out.append(f2.readlines())
out2 = ''
for f in out:
    out2 += f
out2.replace('//t', '')
out2.replace('//m', '')
out2.replace('//', '')

