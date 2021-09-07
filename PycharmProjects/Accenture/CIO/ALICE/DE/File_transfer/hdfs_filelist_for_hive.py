# -*- coding:utf-8 -*-
from hdfs.ext.kerberos import KerberosClient
import os
import datetime
import tempfile
import pandas as pd
import sys

hdfs_parent_path = '/data/insight/cio/alice/demo/contracts_files/'
hdfs_upload_path = '/data/insight/cio/alice.pp/hivetable/documents_name'

# get date folder with the yesterday
date_str = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y%m%d')
hdfs_path = os.path.join(hdfs_parent_path, date_str)

def put_file_to_hdfs():
    # Get whole files in the hdfs path
    try:
        # init the client
        client = KerberosClient("http://name-node.cioprd.local:50070")

        # in case the date folde not exits in the HDFS folder, exit with the current code without running following step
        if date_str not in client.list(hdfs_parent_path):
            return True

        file_list = client.list(hdfs_path)

        # remove the .txt extensions
        file_list = [x[:-4] for x in file_list]

        # here is to make a temperate folder in the server side, and put the files to the folder
        tmp_folder = tempfile.mkdtemp()
        df = pd.DataFrame(file_list)
        df.columns = ['documentname']
        df['dt'] = date_str

        file_name = date_str + '.csv'
        file_path = os.path.join(tmp_folder, file_name)
        # ensure even in the temperate folder, the files shouldn't exit
        if os.path.exists(file_path):
            os.remove(file_path)
        df.to_csv(file_path, index=False, header=False)

        # Here I want to ensure that the result file in the destination folder should be removed if rerun the job
        if date_str in client.list(hdfs_upload_path):
            try:
                client.delete(os.path.join(hdfs_upload_path, file_name))
            except IOError as e:
                print("HDFS folder %s dosen't exits!"%(file_name))

        client.upload(hdfs_upload_path, file_path, overwrite=True)

        # after the program finishs, then remove the file in the temperate folder
        try:
            os.remove(file_path)
        except IOError as e:
            print("couldn't remove the files the in the temperate folder: {}".format(file_path))

        print('Whole step has finished for date: %s' % (date_str))
    except Exception as e:
        print("HDFS folder {} donen't exits!".format(hdfs_path))


if __name__ == "__main__":
    put_file_to_hdfs()

