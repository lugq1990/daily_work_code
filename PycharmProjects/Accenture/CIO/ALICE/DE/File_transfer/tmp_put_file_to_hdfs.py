# -*- coding:utf-8 -*-
import os
sftp_path = '/sftp/cio.alice'
path = '/sftp/cio.alice/20190603'
folder_list = ['data_0',  'data_1',  'data_2',  'data_3',  'data_4',  'data_5',  'data_6',  'data_7',  'data_8',  'data_9',
               'data_A',  'data_B',  'data_C',  'data_D',  'data_E',  'data_F',  'data_G',  'data_H',
               'data_I',  'data_J',  'data_K',  'data_L',  'data_M',  'data_N',  'data_O',  'data_P',
               'data_Q',  'data_R',  'data_S',  'data_T',  'data_U',  'data_V',  'data_W',  'data_X',
               'data_Y',  'data_Z', 'others']

# make dir
for f in folder_list:
    try:
        os.mkdir(os.path.join(path, f))
    except Exception as e:
        pass

# after make dir, move files from parent path to sub-folder
# import shutil
# file_list = [x for x in os.listdir(sftp_path) if x.endswith('.txt')]
# for f in file_list:
#     sub_folder = 'data_' + f[-5].upper()
#     if sub_folder not in folder_list:
#         sub_folder = folder_list[-1]
#     shutil.copyfile(os.path.join(sftp_path, f), os.path.join(os.path.join(path, sub_folder), f))
#     # os.rename(os.path.join(sftp_path, f), os.path.join(os.path.join(path, sub_folder), f))

# for fo in folder_list:
#     print('Now is folder:', fo)
#     if len(os.listdir(os.path.join(path, fo))) == 0:
#         continue
#     os.system('hdfs dfs -put -f %s/*.txt /data/insight/cio/alice/contracts_files/20190603/ '%(os.path.join(path, fo)))

print('Now is for whole folder!')
for fo in folder_list:
    print('now is folder: ', fo)
    if len(os.listdir(os.path.join(path, fo))) == 0:
        continue
    os.system('hdfs dfs -put -f %s/*.txt /data/insight/cio/alice/contracts_files/whole_files2/ ' % (os.path.join(path, fo)))
