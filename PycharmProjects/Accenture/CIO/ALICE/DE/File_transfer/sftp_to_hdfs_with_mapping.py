# -*- coding:utf-8 -*-
"""This module is used to put the files to HDFS and move the files in s3 delta folder
to the whole files folder in S3 bucket."""

import paramiko
from paramiko import AuthenticationException
from paramiko import SFTPError
import logging
import os
from hdfs.ext.kerberos import KerberosClient
import time
import sys
import configparser
from inspect import getsourcefile
import boto3
from boto3.exceptions import ResourceLoadException
from hdfs import HdfsError


# sftp_path = '/sftp/cio.alice'
sftp_path = '/sftp/cio.alice/duplicate_test'
hdfs_parent_path = '/data/insight/cio/alice.pp/contracts_files'
hdfs_whole_path = os.path.join(hdfs_parent_path, 'whole_files2')
hdfs_mapping_path = os.path.join(hdfs_parent_path, 'mapping_files')
hdfs_zip_file_path = os.path.join(hdfs_parent_path, 'zip_to_delete')

mapping_sftp_path = '/'.join([sftp_path, 'tmp_files'])

retry_times = 3   # how many times to retry


def init_logger():
    """
    This is just to init the logger for later step use case.
    :return: logger object
    """
    logger = logging.getLogger('sftp_to_hdfs')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y%m%d %H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


# first should init the logger object.
logger = init_logger()


# using the ssh to execute the command
def execute_command(ssh, command):
    """
    Useful function to execute the command for remote server with ssh client.
    Also ensure only if the command execute finished.
    :param ssh: init ssh client
    :param command: needed to execute command
    :return: True
    """
    # logger.info('Now is: ', command)
    stdin, stdout, stderr = ssh.exec_command(command)
    stdout.readlines()   # ensure the the step have to finish!
    return True


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

    logger.info('Start to init the SSH client!')
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
            logger.error("couldn't init the SSH with %d times"%(cur_step))

    # if after many time to init with error, just raise one error to the system
    if ssh is None or sftp is None:
        sys.exit("couldn't init the SSH object for %d times" % (retry_times))

    return ssh, sftp


# Add init HDFS client retry function
def init_client():
    """
    Used to init the hdfs client for interacting with HDFS like querying alike with
    hdfs operation. Also with retry logic. If init failure with 3 times, will raise error.
    :return: init hdfs client object.
    """
    statisfy = False
    cur_step = 0
    while not statisfy:
        try:
            client = KerberosClient("http://name-node.cioprd.local:50070")
            cur_step += 1
            if client is not None or cur_step > retry_times:
                statisfy = True
        except:
            time.sleep(60)
            logger.error("Couldn't init the HDFS client with %d times" % cur_step)

    if client is None:
        sys.exit("Couldn't init the client object for %d times" % retry_times)

    return client


# Here I just make one function to put the file and some move step
def move_step(ssh, client, mapping_file, sftp):
    """
    The moving main logic to do file transfer only for single mapping file.
    Logic is:
    1. get the mapping file name with date string.
    2. make the date folder with date string from mapping in remote server and HDFS.
    3. open the mapping file and copy the files in mapping file from SFTP server to date folder
    4. put the files into HDFS and also put the mapping file to HDFS for later use case.
    :param ssh: init ssh client for remote server
    :param client: init hdfs client for interacting with HDFS
    :param mapping_file: the needed mapping file just with name
    :return: True if successfully otherwise with error.
    """
    # Just to ensure, not needed in fact.
    if ssh is None or client is None:
        return 'When init the ssh connection with error, ' \
               'if want to run the following step, have to make re-init ssh'

    # Here is to get the date for folder in HDFS and SFTP
    date_str = mapping_file.split('_')[1].replace('.TXT', '')
    logger.info('Get Date: %s' % date_str)

    # here is to check for the date should be like 20190101
    def _check_date(date_str):
        try:
            int(date_str)
        except TypeError as e:
            logger.error('Get wrong date type with %s' % date_str)

        if not (int(date_str[:4]) < 2900 and int(date_str[4:6]) < 13 and int(date_str[6:]) < 32):
            logger.error('You have to give the correct date! Given is %s' % date_str)

    # start checking step
    _check_date(date_str)


    # Here for there will be one scenario that maybe there will be multiple mapping file with same day
    # but with different timestamp, so here should adjust the logic to make date folder in SFTP and HDFS date folder
    # here is to judge whether the date folder exits in SFTP and HDFS

    # first step is to create a date folder in HDFS
    hdfs_date_path = '/'.join([hdfs_parent_path, date_str])
    sftp_date_path = '/'.join([sftp_path, date_str])
    create_folder_command = "mkdir -p %s && hdfs dfs -mkdir -p %s " % (sftp_date_path, hdfs_date_path)
    # execute the create folder command
    execute_command(ssh, create_folder_command)
    # to check whether the date folder in HDFS has already created
    try:
        time.sleep(1)
        if not client.list(hdfs_date_path):
            execute_command(ssh, create_folder_command)  # in case folder not created!
    except Exception as e:
        pass

    mapping_file_path = os.path.join(sftp_path, mapping_file)

    logger.info('Copy contracts of mapping file: %s to %s'%(mapping_file, sftp_date_path))

    move_file_command = """
        awk '{print $1}' %s | while read num
        do
        cp %s/$num.txt %s
        done 
        """ % (mapping_file_path, sftp_path, sftp_date_path)

    # start to move the files to another SFTP folder for later step to move the files!
    execute_command(ssh, move_file_command)


    # as Here I have move the files to one date folder, I could use the command to put the whole files to HDFS
    # here I should move the mapping files to another folder for later step use case
    # after the whole step finishs, should follow these steps: 1. move mapping files to another folder,
    # 2. remove the whole .txt files

    logger.info('Start to put the files to HDFS path: %s and move the mapping files to %s' % (hdfs_date_path, mapping_sftp_path))

    ###  In case if there are too many files in the SFTP folder, then here could continou to put files to HDFS
    # get how many files in SFTP
    try:
        files_list_in_sftp = [x for x in sftp.listdir(sftp_path)
                              if x.endswith('.txt') and not x.lower().startswith('mapping')]
    except Exception as e:
        logger.warning("When to get the files in SFTP with error: %s" % e)

    # if there isn't any files in the sftp folder, here should pass without any action
    if len(files_list_in_sftp) == 0:
        if mapping_file is None:
            logger.error("Get mapping file: %s but without any files in SFTP folder: %s" % (mapping_file, sftp_path))
        else:
            logger.warning("There isn't any txt files in %s!")
        return None

    if len(files_list_in_sftp) > 100000:
        # here should also put these files to the whole files folder just like production scenario.
        # here also put the mapping file to HDFS for later needed to recompute.
        put_hdfs_command = """
            hdfs dfs -put -f %s/*.txt %s && 
            cp %s %s/ && 
            hdfs dfs -put -f %s/*.txt %s && 
            hdfs dfs -put -f %s/*.txt %s 
            """ % (sftp_date_path, hdfs_date_path, mapping_file_path,
                   mapping_sftp_path, sftp_date_path, hdfs_whole_path,
                   mapping_sftp_path, hdfs_mapping_path)
        # start to put files and move the mapping file!
        execute_command(ssh, put_hdfs_command)
    else:
        logger.info("Now use HDFS client to put files to HDFS!")
        # there are too many files in sftp, have to put one by one
        put_files_to_hdfs_with_client(client, sftp, mapping_file)

    logger.info('For mapping file: %s has finished' % mapping_file)
    return True


def put_files_to_hdfs_with_client(client, sftp, mapping_file):
    """
    In case there are too many files in the sftp folder, we could put files to HDFS by
    opening the files in SFTP and write the files to HDFS, also with move mapping files
    from sftp parent folder to SFTP mapping folder and put it to HDFS mapping folder
    :param client: HDFS instanced client object to interact with HDFS
    :param sftp: SFTP object to interact with remote server SFTP folder
    :param mapping_file: which mapping file should be processed.
    :return: True if without error!
    """
    # first get date string from the mapping file name string
    date_str = mapping_file.split('_')[1].replace('.TXT', '')

    # The path of SFTP and HDFS path for files
    # as we have already moved the mapping file to date folder
    sftp_date_path = '/'.join([sftp_path, date_str])
    hdfs_date_path = '/'.join([hdfs_parent_path, date_str])

    # then we should get the mapping date folder with whole files list,
    # in case there are many one date folder for mapping files, so here I first get
    # HDFS date folder files list, then get the only hasn't been putted files
    # for later iteration
    try:
        # in case the date folder not exits
        already_put_hdfs_list = client.list(hdfs_date_path)
    except:
        pass
    filelist = [x for x in sftp.listdir(sftp_date_path) if x.endswith('.txt')]
    filelist = list(set(filelist) - set(already_put_hdfs_list))

    logger.info("When put files to HDFS, get date folder : %s with files: %d haven't been putted!" %
                (sftp_date_path, len(filelist)))

    # using the client object to put files to HDFS
    curr_step = 1
    whole_step_finshed = False
    while not whole_step_finshed:
        try:
            # first open the sftp files and put the files to `date` HDFS folder and `whole_files2` folder
            for file in filelist:
                with sftp.open(os.path.join(sftp_date_path, file), 'r') as f:
                    data = f.read()
                    # write the data to HDFS
                    client.write(os.path.join(hdfs_date_path, file), data=data, overwrite=True)
                    client.write(os.path.join(hdfs_whole_path, file), data=data, overwrite=True)

                if filelist.index(file) % 5000 == 0:
                    logger.info("Already put %d files to HDFS." % filelist.index(file))

            # Here I also should move the mapping file to mapping file folder in SFTP
            # as sftp supposes that for file moving step, the file shouldn't exist.
            try:
                sftp.rename(os.path.join(sftp_path, mapping_file), os.path.join(mapping_sftp_path, mapping_file))
            except Exception as e:
                logger.warning("When rename the mapping file to Mapping folder, mapping file already exits.")
            # put the mapping files to HDFS
            logger.info("Put mapping file %s to HDFS Mapping path: %s" % (mapping_file, hdfs_mapping_path))
            with sftp.open(os.path.join(mapping_sftp_path, mapping_file), 'r') as f:
                data = f.read()
                client.write(os.path.join(hdfs_mapping_path, mapping_file), data=data, overwrite=True)

            if curr_step <= retry_times:
                whole_step_finshed = True
        except Exception as e:
            logger.warning("When open the sftp files and put to HDFS with error %s " % e)
            # retry
            curr_step += 1
            time.sleep(60)
    return True


# This function is used to copy one folder files to another folder, after copy step finished,
# then just remove the whole files in the source folder
def move_folder_to_folder_for_s3(config):
    """
    This is to move the files from s3 source folder to s3 destination folder.
    :param config: config object contains the s3 useful info
    :return: True if moving step finishes otherwise will raise error.
    """
    ### now is not to use the config result, manually set with these parameters
    # access_key = config['config']['access_key']
    # secret_key = config['config']['secret_key']
    # bucket_name = config['config']['bucket_name']
    # src_folder_name = config['config']['src_folder_name']
    # des_folder_name = config['config']['des_folder_name']
    access_key = 'AKIAJESPISZ2QBVPQFGQ'
    secret_key = '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'
    bucket_name = '30899-aliceportal-stage'
    src_folder_name = 'Delta2'
    des_folder_name = 'TextFiles'

    logger.info('Start to move files in S3 bucket.')

    try:
        s3 = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key).resource('s3')
        my_bucket = s3.Bucket(bucket_name)

        # first to get whole files in s3 bucket
        file_list = []
        for f in my_bucket.objects.filter(Prefix=src_folder_name + '/'):
            if f.key.endswith('.txt'):
                file_list.append(f.key)

        # check for the files received
        logger.info("Get %d files in the source S3 bucket of folder %s " % (len(file_list), src_folder_name))

        # ['/Detla/a.txt'] - > ['/TextFiles/a.txt']
        # use this code to copy files from one folder to another folder, change the desc file prefix
        des_file_list = [x.replace(src_folder_name, des_folder_name) for x in file_list]

        # 2 steps: 1. copy files from source folder ->desc folder
        for i in range(len(file_list)):
            src_dirc = {'Bucket': bucket_name, 'Key': file_list[i]}
            s3.meta.client.copy(src_dirc, bucket_name, des_file_list[i])

        # Then here is to remove the files in the source part,
        # for now shouldn't used as if here delete the files,
        for f in file_list:
            s3.Object(bucket_name, f).delete()

    except ResourceLoadException as e:
        logger.error('Connect to S3 with failure: %s' % e)

    logger.info('S3 step has finished!')
    return True


def create_detete_flag_file(client, sftp, mapping_file_list):
    """
    This is just to create or append one file with new zip records in HDFS folder,
    as there maybe many mapping files in the sftp folder, so here just combine whole
    mapping files contents with ZIP extension!
    :param client: hdfs client to create or modify the file in HDFS
    :param sftp: remote server to open mapping files
    :param mapping_file_list: mapping file list that are needed to get
    :return: True if whole step finished without error
    """
    content_list = []

    if len(mapping_file_list) == 0:
        logger.warning("There isn't any mapping file in HDFS")
        return True

    def _get_mapping_content(mapping_file_name, content_list):
        """
        here is private function to open the mapping file and get the full paths,
        also add with retry logic.
        :param mapping_file_name: which mapping file to process
        :param content_list: whole content list with previous mapping file contents.
        :return: changed content_list
        """
        satisfy = False
        cur_step = 0
        while not satisfy or cur_step > retry_times:
            try:
                with sftp.open(os.path.join(sftp_path, mapping_file_name), 'r') as f:
                    data = f.readlines()
                    # without _00*.
                    data = [x.split('\t')[1].split('_')[0] for x in data if x.split('\t')[2].upper() == 'ZIP']

                if len(data) != 0:
                    content_list.extend(data)

                # no matter there is content with zip or not, if without error,
                # then just change satisfy to True!
                satisfy = True
            except SFTPError as e:
                logger.warning("When open SFTP mapping file with error %s with %d times!" % (e, cur_step))
                time.sleep(60)
                cur_step += 1

        if cur_step > retry_times:
            logging.error("Even with %d times, couldn't get mapping contents!" % retry_times)
            raise SFTPError("Even with %d times, couldn't get mapping contents! Please rerun the code!" % retry_times)

        return content_list

    # loop for the whole mapping files
    for file_name in mapping_file_list:
        content_list = _get_mapping_content(file_name, content_list)

    # should return unique data
    content_list = list(set(content_list))

    write_data = '\n'.join(content_list)

    if len(content_list) == 0:
        logger.info("There isn't ZIP records in the mapping file!")
        return True
    else:
        logger.info("Write the zip content to HDFS files with path: %s" % hdfs_zip_file_path)
        # first to check whether the file exists
        zip_txt_file_name = 'zip_delete.txt'
        satisfy = False
        cur_step = 0
        while not satisfy or cur_step > retry_times:
            try:
                if len(client.list(hdfs_zip_file_path)) == 0:
                    logger.info("HDFS file for zip has been removed!")
                    # file has been deleted
                    client.write(hdfs_path=os.path.join(hdfs_zip_file_path, zip_txt_file_name), data=write_data)
                else:
                    logger.info("HDFS file for zip use append mode.")
                    # if the file already exists, then should first add data with \n for new added data
                    write_data = '\n' + write_data
                    # file exists, so just append the data to original file
                    client.write(hdfs_path=os.path.join(hdfs_zip_file_path, zip_txt_file_name), data=write_data, append=True)

                satisfy = True
            except HdfsError as e:
                logger.error("When write data to HDFS with path: %s with error: %s" % (hdfs_zip_file_path, e))
                cur_step += 1
                time.sleep(60)

        if cur_step > retry_times:
            logger.error("When write data to HDFS with %d times, but also with error!" % cur_step)
            raise HdfsError("When write data to HDFS with %d times, but also with error!" % cur_step)

        return True


def get_mapping_file(sftp_c):
    """
    get the mapping file list in sftp server!
    :param sftp_c: sftp client to interact
    :return: mapping files list
    """
    if sftp_c is None:
        logger.error("When to get mapping file list from sftp server, given Null object sftp client!")
        raise SFTPError("When to get mapping file list from sftp server, given Null object sftp client!")

    # first get how many mapping file
    # Here to get the mapping files list with using sftp, should combined with try-catch,
    # as maybe there will raise error! if error the retry!
    get_mappings = False
    cur_step = 0
    while not get_mappings or cur_step > retry_times:
        try:
            mapping_file_lists = [x for x in sftp_c.listdir(sftp_path) if x.lower().startswith('mapping')]
            get_mappings = True
        except SFTPError as e:
            logger.error("When to list mapping files in %s with error: %s" % (sftp_path, e))
            cur_step += 1
            time.sleep(60)

    if mapping_file_lists is None:
        raise SFTPError("When to get mapping file list, with retry %d times, doesn't get either!" % cur_step)

    return mapping_file_lists


if __name__ == '__main__':
    # as there isn't any try-catch, the reason is that for no matter which function
    # faces with error, then should just raise error from each function!

    # This is to get the config content
    config_path = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))
    config_name = 'config.conf'

    config = configparser.ConfigParser()
    if config_name not in os.listdir(config_path):
        raise IOError("Doesn't find the config file: %s in %s!" % (config_name, config_path))

    config.read(os.path.join(config_path, config_name))

    client = init_client()

    ssh, sftp = init_ssh(config)

    mapping_file_list = get_mapping_file(sftp)

    # as here just add with zip file for deleting logic, should execute this function first
    logger.info("Make txt file for ZIP to HDFS.")
    create_detete_flag_file(client, sftp, mapping_file_list)

    # according to different number mapping files to do different step
    if len(mapping_file_list) == 0:
        logger.warning("This isn't any mapping file in SFTP folder: %s" % sftp_path)
    elif len(mapping_file_list) == 1:
        move_step(ssh, client, mapping_file_list[0], sftp)
    else:  # there are many mapping files
        for mapping_file in mapping_file_list:
            move_step(ssh, client, mapping_file, sftp)

    # After put files with mapping file finish, then with the s3 bucket folder move step
    move_folder_to_folder_for_s3(config)

    logger.info("Whole step finished successfully!")
