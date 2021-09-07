# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""
import zipfile
import os
import shutil
local_path = '.'
file_name = 'ModelTraining.zip'

folder_name = file_name.split('.')[0]

try:
    print("Remove folder")
    folder_name = file_name.split('.')[0]
    shutil.rmtree(os.path.join(local_path, folder_name))
    os.mkdir(os.path.join(local_path, folder_name))
except:
    pass

zip = zipfile.ZipFile(os.path.join(local_path, file_name))
zip.extractall(path=os.path.join(local_path, folder_name))
print("finished.")