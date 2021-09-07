# -*- coding:utf-8 -*-
import cv2
import numpy as np
import shutil
import glob
import os

### This function is used to compute one directory all image shape,
### Used to get different image information
def compute_mean_width(path):
    file_list = glob.glob1(path, "*.jpg")

    shape_list = []
    # Loop for all files
    for f in file_list:
        shape_list.append(cv2.imread(os.path.join(path, f)).shape)
    return shape_list


if __name__ == "__main__":
    path = "C:/Users/guangqiiang.lu/Documents/lugq/github/cptn/bpo_images/all_images"
    shape_list = compute_mean_width(path)
    height_mean = np.mean([h[0] for h in shape_list])
    print('This directory image mean width is {}'.format(height_mean) )