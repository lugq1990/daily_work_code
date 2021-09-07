# -*- coding:utf-8 -*-
"""Here is not only to list the files in the HDFS,
also have to get the filename in the SFTP folder mapping file, our goal is to put the files to the
hdfs hive external table path"""

import pandas as pd
from paramiko import AuthenticationException
import logging
import numpy as np
import configparser
from inspect import getsourcefile
from hdfs.ext.kerberos import KerberosClient
from hdfs import HdfsError
import paramiko
import os
import tempfile
import shutil
import time
import sys


logger = logging.getLogger('hdfs_to_hive')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y%m%d %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)

# hdfs_parent_path = '/data/insight/cio/alice/contracts_files/'
# hdfs_upload_path = '/data/insight/cio/alice/hivetable/documents_name'
# hdfs_mapping_path = os.path.join(hdfs_parent_path, 'mapping_files')
# sftp_mapping_file_path = '/sftp/cio.alice/files_of_mapping_sftp'

hdfs_parent_path = '/data/insight/cio/alice.pp/contracts_files'
hdfs_upload_path = '/data/insight/cio/alice.pp/hivetable/documents_name_zip'
hdfs_mapping_path = os.path.join(hdfs_parent_path, 'mapping_files')
sftp_mapping_file_path = '/sftp/cio.alice/duplicate_test'
# this should the mapping file sftp path.
sftp_mapping_file_path = os.path.join(sftp_mapping_file_path, 'tmp_files')

retry_times = 3   # how many times to retry
is_new_coming_mapping = True


# first to init the ssh, here also add some functionality with retry
def init_ssh(config):
    """
    This is used to init the remote ssh client and sftp object for later step use case.
    Here also add with the retry step for init the ssh with retry_times, for now is 3 times.
    Any time with exception will wait for 60 seconds to retry. If retry with 3 times, but
    also couldn't init the ssh, will raise one error.
    :param config: config object that contains useful info to init the ssh and sftp
    :return: inited ssh and sftp object
    """
    host = config['config']['host']
    port = int(config['config']['port'])
    username = config['config']['username']
    password = config['config']['password']

    # cause sometimes I find that this function will return None that couldn't init the SSH,
    # Here I add retry another time
    satisfy = False
    cur_step = 0
    while not satisfy:
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            ssh.connect(host, port=port, username=username, password=password, look_for_keys=False)
            sftp = ssh.open_sftp()
            cur_step += 1
            if (ssh is not None and sftp is not None) or cur_step == retry_times:
                satisfy = True
        except AuthenticationException:
            time.sleep(60)
            logger.warning("couldn't init the SSH with %d times" % cur_step)
    if ssh is None or sftp is None:
        logger.error("When init the SSH with error, couldn't init the SSH with %d times!" % retry_times)
        sys.exit("couldn't init the SSH object for %d times" % retry_times)

    return ssh, sftp


# init hdfs client, also with retry
def init_clinet():
    """
    Used to init the hdfs client for interacting with HDFS like querying alike with
    hdfs operation. Also with retry logic. If init failure with 3 times, will raise error.
    :return: init hdfs client object.
    """
    statisfy = False
    cur_step = 0
    while not statisfy:
        try:
            client = KerberosClient("http://name-node.cioprd.local:50070;http://name-node2.cioprd.local:50070")
            cur_step += 1
            if client is not None or cur_step > retry_times:
                statisfy = True
        except Exception as e:
            if client is None:
                time.sleep(60)
                logger.warning("Couldn't init client with %d times" % cur_step)
    if client is None:
        logger.error("Even with %d times retry, also couldn't init the client." % retry_times)
        sys.exit("Even with %d times retry, also couldn't init the client." % retry_times)

    return client


# this is to get the hdfs files
def _get_hdfs_files(client, date_folder):
    """
    To get the hdfs files list with hdfs client for needed date string as date folder in HDFS.
    :param client: hdfs client object
    :param date_folder: Date string like '20190101'
    :return: without extension file array for efficient use case.
    """
    hdfs_date_path = '/'.join([hdfs_parent_path, date_folder])
    try:
        # in case the date folde not exits in the HDFS folder, exit with the current code without running following step
        if date_folder not in client.list(hdfs_parent_path):
            logger.error("The date folder in HDFS: %s does not exit!"%(date_folder))
            return None

        logger.info('Start to get the hdfs files!')
        file_list = client.list(hdfs_date_path)
        # remove the .txt extensions
        file_list = [x[:-4] for x in file_list]

        return np.array(file_list)
    except Exception as e:
        logger.error('When to list the files in HDSF with error! %s' % e)
        return None


# to get the file name from remote mapping files list for later step
def get_mapping_files_list(sftp):
    """
    Get the mapping file list with init sftp client, just convert the string to lower and
    starting with 'mapping'
    :param sftp: init sftp client
    :return: if without any mapping with None, otherwise with mapping file list
    """
    # cause there maybe many mapping file in the folder, first is to list the folder with the mapping files
    getted_mapping_files = [x for x in sftp.listdir(sftp_mapping_file_path) if x.lower().startswith('mapping')]
    if len(getted_mapping_files) == 0:
        return None
    else:
        return getted_mapping_files


# get the mapping files content
def _get_mapping_content(mapping_file_list, sftp):
    """
    Get the contents of mapping file, but with append with different mapping files.
    :param mapping_file_list: which mapping file list to be processed, if there are many same date mapping file,
        then will open the files and combine with each other to output to make the same mapping content.
    :param sftp: sftp for open the mapping file
    :return: merged mapping file contents as numpy array for later DataFrame join.
    """
    contents_list = []

    # 2019/09/03 add: there would be one more mapping file for both zip and regular mapping file
    # we should handle both cases here, for the regular mapping file will be end with:Regular.txt,
    # for zip mapping file will be end with: Zip.txt
    for fn in mapping_file_list:
        mapping_file_path = '/'.join([sftp_mapping_file_path, fn])

        try:
            logger.info("start to get the contents of mapping file: %s" % fn)
            with sftp.open(mapping_file_path, 'r') as f:
                records_list = [x.replace('\n', '').strip().split('\t') for x in f.readlines()]
                if not fn.replace('.txt', '').lower().endswith('zip'):
                    # if this is a regular mapping, then here I should add another two columns
                    # to the content_list result, so that we could just get the result as the same way
                    # like zip mapping file
                    logger.info("Get regular mapping file: %s" % fn)
                    # remove the last column with date value
                    records_list = [x[:-1] for x in records_list]
                    none_list = [None, None]
                    records_list = [x + none_list for x in records_list]
                else:
                    logger.info("Get zip mapping file: %s" % fn)
                    # zip mapping data should remove last columns with date before combine,
                    # otherwise, will be different shape with regular mapping file
                    records_list = [x[:-1] for x in records_list]
                contents_list.extend(records_list)
        except Exception as e:
            logger.error("When to get the mapping file %s, with error %s" % (file_name, e))
            return None

    # here add with the extension column, also with real name
    contents_array = np.asanyarray(contents_list)
    if contents_array.shape[1] != 4:
        logger.error("Get wrong shape data {} dimension, should be 4 dimension".format(contents_array.shape[1]))
    return contents_array


# Here just open the remote mapping file and combine the mapping file and HDFS files
def _merge_mapping_hdfs_files(mapping_file_list, sftp, client):
    """
    Get the content from the mapping file list and merge mapping files
    to be one DataFrame and join with HDFS DataFrame
    :param mapping_file_list: which mapping files list to be opened and merge
    :param sftp: sftp for open file
    :param client: client for list HDFS folder
    :return: merged DataFrame
    """
    date_str = mapping_file_list[0].split('_')[1].replace('.TXT', '')
    # first get the HDFS files list
    hdfs_file_array = _get_hdfs_files(client=client, date_folder=date_str)

    # get the mapping files contents
    mapping_contents_array = _get_mapping_content(mapping_file_list=mapping_file_list, sftp=sftp)

    if hdfs_file_array is None:
        logger.error('Does not get file from HDFS! please check with the HDFS data')
        return None
    if mapping_contents_array is None:
        logging.error('Dose not get contents from mapping files')
        return None

    # after get the both data from the mapping file and HDFS files,
    # I make two DataFrame to represent different data, so that I could join two different DataFrame
    hdfs_df = pd.DataFrame(hdfs_file_array, columns=['file_name'])
    # here also add the DHFS path of the file list, just add with the hdfs_parent_path with date folder
    hdfs_df['hdfs_path'] = '/'.join([hdfs_parent_path, date_str])

    # cause for now, according to mapping file contents with different order, here is just to judge
    # as with different order, the merged dataframe is None
    if is_new_coming_mapping:
        mapping_df = pd.DataFrame(mapping_contents_array, columns=['file_name', 'file_path', 'extension', 'real_name'])
    else:
        mapping_df = pd.DataFrame(mapping_contents_array, columns=['file_path', 'file_name', 'extension', 'real_name'])

    # cause just with simple merge, here couldn't make that, here I convert the data type and merge
    hdfs_df['file_name'] = hdfs_df['file_name'].astype(str)
    mapping_df['file_name'] = mapping_df['file_name'].astype(str)
    hdfs_df = hdfs_df.set_index('file_name')
    mapping_df = mapping_df.set_index('file_name')

    # this dosen't work!
    # merge_df = pd.DataFrame.join(hdfs_df, mapping_df, how='inner', on='file_name').reset_index()
    merge_df = hdfs_df.join(mapping_df, on='file_name', how='inner').reset_index()

    # Here I should also add the date columns according to the mapping file date
    merge_df['date'] = date_str

    if len(merge_df) == 0:
        logger.info("there are %d rows in mapping dataframe" % len(mapping_df))
        logger.info("there are %d rows in hdfs dataframe" % len(hdfs_df))
        logger.warning("Dosen't get any files merged!")

    # previous df column
    # merge_df = merge_df[['file_name', 'hdfs_path', 'file_path', 'date']]

    # here add the extension with last column
    merge_df = merge_df[['file_name', 'hdfs_path', 'file_path', 'date', 'extension', 'real_name']]

    return merge_df


# This will create one temperate folder in system, and put the merged DataFrame to HDFS
# here I put different files to HDFS with different date
def put_merged_df_to_hdfs(client, sftp, mapping_file_list):
    """
    Put the merged DataFrame to HDFS for HIVE
    :param client: client server
    :param sftp: sftp for open files
    :param mapping_file_list:
        Here is a list for that date, if the with just one, doesn't matter, but also is a list.
    :return: None, finished put files
    """
    date_str = mapping_file_list[0].split('_')[1].replace('.TXT', '')
    tmp_folder = tempfile.mkdtemp()
    date_str = date_str + '.csv'
    file_path = os.path.join(tmp_folder, date_str)

    # get merged df
    merged_df = _merge_mapping_hdfs_files(mapping_file_list=mapping_file_list, sftp=sftp, client=client)

    merged_df.to_csv(file_path, index=False, header=False)

    # Here I want to ensure that the result file in the destination folder should be removed if rerun the job
    if date_str in client.list(hdfs_upload_path):
        try:
            client.delete(os.path.join(hdfs_upload_path, date_str))
        except FileNotFoundError as e:
            print("file %s not in HDFS path, couldn't be removed" % date_str)

    logger.info('Start to put %s file to hdfs path: %s' % (date_str, hdfs_upload_path))
    client.upload(hdfs_upload_path, file_path, overwrite=True)

    # after the program finishs, then remove the file in the temperate folder
    try:
        os.remove(file_path)
    except IOError as e:
        logger.error("couldn't remove the files %s the in the "
                     "temperate folder with error %s!" % (date_str, e))


def remove_remote_mapping_file(sftp):
    """
    Here add the logic to remove the mapping files if the whole step finished without error.
    just change with removing the whole mapping files
    :param sftp: sftp object for remove use case
    :param file_list: mapping files list in remove server
    :return: Nothing to return
    """
    try:
        file_list = sftp.listdir(sftp_mapping_file_path)
        for f in file_list:
            sftp.remove(os.path.join(sftp_mapping_file_path, f))
    except AuthenticationException as e:
        logger.error('when to remove the mapping file with error %s ' % e)


if __name__ == "__main__":
    # This is to get the config content
    config_path = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))
    config_name = 'config.conf'

    config = configparser.ConfigParser()
    config.read(os.path.join(config_path, config_name))

    ssh, sftp = init_ssh(config)

    client = init_clinet()

    mapping_file_list = get_mapping_files_list(sftp)

    # check whether the mapping file exits, if not, just pass the whole bellow step
    if mapping_file_list is None:
        logger.warning("There isn't any mapping file in the SFTP server side.")
        pass
    else:
        # Here for there is one scenario that will be 2 mapping file with same date, so here I just
        # convert the mapping_file_list to a dictionary: key with date, value with a list for that with
        # same date, so that I could merge the mapping content wth same date with different timestamp
        from collections import defaultdict
        mapping_file_dict = defaultdict(list)
        unique_date = list(set([x.split('_')[1] for x in mapping_file_list]))
        for date in unique_date:
            for file_name in mapping_file_list:
                if date in file_name:
                    mapping_file_dict[date].append(file_name)

        if len(mapping_file_dict) == 0:
            logger.warning("There isn't any mapping files!")
        else:
            for date_mapping in mapping_file_dict.values():
                put_merged_df_to_hdfs(client, sftp, date_mapping)

        # after the whole processing step finished, here should upload mapping file to HDFS
        logger.info("Upload mapping files to HDFS path: %s " % hdfs_mapping_path)
        # if the whole step finish, then should remove the mapping files
        # as next day, we will try to get that folder to find the mapping file
        # to process.
        logger.info('Remove the mapping files in remote server.')
        remove_remote_mapping_file(sftp)

    logger.info('Whole step finished successfully!')
