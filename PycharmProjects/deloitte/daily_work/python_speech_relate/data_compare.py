"""Just load 2 files and compare different rows. 
There are 2 types logic: 
    1. Hash and compare;
    2. difflib to get `-` and `+`
"""

from hashlib import md5

from more_itertools import first
from sympy import sec
path1 = ""
path2 = ""

def read_data(path):
    res = {}
    with open(path, 'r', encoding='utf-8', newline='') as f:
        data = f.readlines()
        for x in data:
            res[md5(x.encode('utf-8')).hexdigest()] = x
    return res

res1 = read_data(path1)
res2 = read_data(path2)

# compare
diff1 = [value for key, value in res1.items() if key not in res2]
diff2 = [value for key, value in res2.items() if key not in res1]


def write_file(data, out_path):
    with open(out_path, 'w', encoding='ISO-8859-1') as f:
        [f.write(x) for x in data]
        
import difflib

def get_diff(data1, data2, return_first_diff=True):
    diff_res = difflib.Differ().compare(data1, data2)
    
    if return_first_diff:
        first_diff = [x for x in diff_res if x.startswith('-')]
        if first_diff:
            return [x.split()[-1] for x in first_diff]
    else:
        second_diff = [x for x in diff_res if x.startswith('+')]
        if second_diff:
            return [x.split()[-1] for x in second_diff]
    return []


diff1 = get_diff(res1, res2)
diff2 = get_diff(res1, res2, return_first_diff=False)