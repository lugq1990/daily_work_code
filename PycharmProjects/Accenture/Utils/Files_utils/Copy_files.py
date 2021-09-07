# -*- coding:utf-8 -*-
import os
import shutil
from fnmatch import fnmatch

"""This function is used to copy one directory all files to another directory
    If the source directory also includes directory, then copy inside files to destination.
    If source directory and target direction is in same directory, 
    then will not copy target directory files"""
def copy_all_files_to_another_dic(src, dst, pattern='*.jpg', silent=True):
    name_index = 0
    dir_list = os.listdir(src)

    # Loop directory to get all directory
    for di in dir_list:
        if di == dst.split('\\')[-1]:
            continue
        if not silent:
            print('Now is in directory: {}'.format(di))
        # Here I use os.walk to get all files
        for p, sub_p, files in os.walk(os.path.join(src, di)):
            for name in files:
                #print(os.path.join(p, name))
                if fnmatch(name, pattern):
                    shutil.copyfile(os.path.join(p, name), os.path.join(dst, str(name_index) + pattern[-4:]))
                name_index += 1


if __name__ == '__main__':
    path = "C:/Users/guangqiiang.lu/Documents/lugq/github/cptn/bpo_images"
    dst_path = 'all_images'
    copy_all_files_to_another_dic(path, os.path.join(path, dst_path), silent=True)