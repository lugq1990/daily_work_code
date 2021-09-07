# import logging
# import os
#
# logger = logging.getLogger('test')
# logger.setLevel(logging.DEBUG)
# path = 'C:/Users/guangqiiang.lu/python_code'
# fh = logging.FileHandler(os.path.join(path, 'test.log'))
# fh.setLevel(logging.DEBUG)
# logger.addHander(fh)
#
# try:
#     1/ 0
# except ZeroDivisionError as e:
#     logger.error('get error with %s'%(e))
import logging
import os
from inspect import getsourcefile
path = 'C:\\Users\\guangqiiang.lu\\python_code'
# logging.basicConfig(filename=os.path.join(path,'pycharm.log'), format='%(asctime)s:%(message)s',
#                     datefmt='%Y-%m-%d %H-%M-%s', level=logging.DEBUG, filemode='w')
# logging.info('start step:')

print('Get:', os.path.dirname(os.path.abspath(getsourcefile(lambda: 0))))

# import tempfile
# try:
#     tmp_path = tempfile.TemporaryDirectory().name()
# except Exception as e:
#     logging.error('something bad happens:%s'%(e))

# import configparser
# config = configparser.ConfigParser()
#
# config.read('config.conf')
# print(config['config']['host'])
# print(type(config['config']['host']))
# print(config['config']['port'])
# print(type(config['config']['port']))


import logging

logger = logging.getLogger('test')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

#create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y%m%d %H:%M:%S')

ch.setFormatter(formatter)

logger.addHandler(ch)

logger.debug('debug message')
logger.info('this is info message')
logger.warning('this is warning')
logger.error('this is error')
logger.critical('this is critical info')




import shutil
import os
import zipfile

abs_path = os.path.abspath(os.curdir)

folder_list = [x for x in os.listdir(abs_path) if os.path.isdir(os.path.join(abs_path, x))]

zipf = zipfile.ZipFile(os.path.join(abs_path,'test.zip'), 'w', zipfile.ZIP_DEFLATED)

os.system('cd %s' % abs_path)

for f in folder_list:
    zipf.write(f)

zipf.close()

print("get files: ", os.listdir(abs_path))

os.system("unzip -d %s %s" % (abs_path, os.path.join(abs_path, 'test.zip')))

print("get files: ", os.listdir(abs_path))


for f in folder_list:
    os.system("rm -rf %s" % os.path.join(abs_path, f))



import zipfile
import os
import tempfile

tmp_path = tempfile.mkdtemp()
abs_path = os.path.abspath(os.curdir)

file_name = [x for x in os.listdir(abs_path) if os.path.isfile(os.path.join(abs_path, x))][0]
zipf = zipfile.ZipFile(os.path.join(abs_path, file_name), 'r')

zipf.extractall(tmp_path)

print("whole files in zip file:", os.listdir(tmp_path))


zipf = zipfile.ZipFile(os.path.join(abs_path, 'test.zip'), 'w')

for folder in [x for x in os.listdir(abs_path) if os.path.isdir(os.path.join(abs_path, x))]:
    for f in os.listdir(os.path.join(abs_path, folder)):
        zipf.write(os.path.join(abs_path, folder, f), f)


import shutil
import os
import tempfile

tmp_path = tempfile.mkdtemp()

abs_path = os.path.abspath(os.curdir)

shutil.make_archive('res', 'zip', root_dir=abs_path, base_dir=abs_path)

try:
    os.mkdir('test')
except:pass

os.system("unzip -d %s %s" % (os.path.join(tmp_path, 'test'), 'res.zip'))

print("files:", os.listdir(tmp_path))


# just try with os.chdir to go to this folder
import os
abs_path = '/anaconda-efs/sharedfiles/projects/mysched_9376/kt_code/mysch/output'
os.chdir(abs_path)

print("We have:", os.listdir(os.curdir))

os.remove(os.path.join(abs_path, 'out.zip'))

import shutil
shutil.make_archive('out', 'zip', abs_path)

import zipfile
import tempfile
tmp_path = tempfile.mkdtemp()

zipf = zipfile.ZipFile(os.path.join(abs_path, 'out.zip'), 'r')
zipf.extractall(tmp_path)

print("whole files:", os.listdir(tmp_path))


# -- this just to remove HDFS folder files with HDFS client
from hdfs.ext.kerberos import KerberosClient
import os

client = KerberosClient("http://name-node.cioprd.local:50070")

hdfs_path = '/data/insight/cio/myschdpp/ai/spark/resources/'

folder_list = client.list(hdfs_path)

for folder in folder_list:
    file_list = client.list(os.path.join(hdfs_path, folder))
    for file in file_list:
        print("now to remove file: %s" % file)
        client.delete(hdfs_path=os.path.join(hdfs_path, folder, file))

for folder in folder_list:
    print("file:", client.list(os.path.join(hdfs_path, folder)))

abs_path = os.path.abspath(os.curdir)

folder_list_local = ['dictionaries', 'models', 'resources']

for i in range(len(folder_list)):
    file_list = os.listdir(os.path.join(abs_path, folder_list_local[i]))
    for f in file_list:
        print("Now to upload %s" % f)
        client.upload(hdfs_path=os.path.join(hdfs_path, folder_list[i]), local_path=os.path.join(abs_path, folder_list_local[i], f))


from gensim.models import LsiModel
