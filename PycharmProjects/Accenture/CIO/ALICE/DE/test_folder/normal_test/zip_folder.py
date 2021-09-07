# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""
import shutil
import os

path = 'C:/Users/guangqiiang.lu/Documents/lugq/github/new_hr_model_tuning/MySched_AIR9376'
folder_name ='ModelTraining'
sub_folder = 'utils'

for f in [x for x in os.listdir(path) if x.endswith('zip')]:
    print("Now to remove", f)
    os.remove(os.path.join(path, f))

##shutil.make_archive('/'.join([path, folder_name, 'package']), 'zip', '/'.join([path, folder_name, sub_folder]))

shutil.make_archive(os.path.join(path, folder_name), 'zip', os.path.join(path, folder_name))
