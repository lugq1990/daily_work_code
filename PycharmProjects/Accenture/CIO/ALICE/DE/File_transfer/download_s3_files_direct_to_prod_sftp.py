# -*- coding:utf-8 -*-
"""This is to implement download file from s3 bucket to memory, then use paramiko to write the data to remote
server for sftp folder"""
import boto3
import paramiko
import os
from paramiko import AuthenticationException
import argparse
import logging
import pandas as pd

logger = logging.getLogger('download_file_SFTP')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y%m%d %H:%M:%S')
ch.setFormatter(formatter)

logger.addHandler(ch)


# sftp_path = '/sftp/cio.alice'
sftp_path = '/sftp/cio.alice/duplicate_test'

access_key = 'AKIAJGK5GH7CU46OSS5Q'
secret_key = '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'
bucket_name = 'aliceportal-30899-prod'
# s3_folder_name = 'Delta'
s3_folder_name = 'Delta_Duplicate_Test'

# this is used to save the already downlaod files
file_path = '/anaconda-efs/sharedfiles/projects/alice_30899/file_transfer_code'

# init the paramiko
# here is just to init the ssh, if want to make it to production server, should use try. catch
def init_sftp():
    host = '10.5.105.51'
    port = 22
    username = 'ngap.app.alice'
    password = 'QWer@#2019'
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        ssh.connect(host, port=port, username=username, password=password, look_for_keys=False)
    except AuthenticationException as e:
        logger.warning('When auth to the SSH, with error: {}'.format(e))
        # if the we couldn't get the connection to SFTP, just stop the program
        pass

    return ssh


# first init boto3
def download_to_sftp(ssh, is_first_time=False):
    # here in fact client is lowest api and resource are high api, they could be just created by session object
    session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    client = session.client('s3')
    s3 = session.resource('s3')
    my_bucket = s3.Bucket(bucket_name)

    # after the init step finish, I should download the files from S3
    # first hive to get the file list, here is just to get how many files in S3
    # this is inner function to save the already download
    # Here I just change the logic, if the file exits, it should be named with 'already_download_file.csv'
    # after with the new coming file has been downloaded, then just modify the save file

    file_name = 'already_download_file.csv'
    def save_downloaded(file_list):
        df = pd.DataFrame(file_list)
        df.columns = ['file_name']
        df.to_csv(os.path.join(file_path, file_name), header=True, index=False)
        return df

    def load_already_download():
        # here is to get the most recent date files
        return pd.read_csv(os.path.join(file_path, file_name))

    # Here I come up with one idea that not to get the already download file from disk files,
    # I could use the Remote server to get the already download files with .txt extension
    # if the job fails, then here should just save the file names to the local folder
    def list_remote_files(ssh):
        with ssh.open_sftp() as sftp:
            file_list = sftp.listdir(sftp_path)
            already_download_list = [x for x in file_list if x.endswith('.txt')]
        return already_download_list

    # already_download_list = []
    # if not is_first_time:
    #     already_download_df = load_already_download()
    #     # if the code is to download the file not with first time, I have to get the files that are already downloaded
    #     # also with the files that are also downloaded to append that to the already dowdload file list
    #     already_download_list = already_download_df['file_name'].values.tolist()

    # not to get files with local files, with the remote server
    already_download_list = list_remote_files(ssh)

    # I have tested, this works!
    # this is just to get the object content, write it into memory, and open remote sftp file and write content into it
    logging.info('Start to download!')
    i = 0
    for f in my_bucket.objects.filter(Prefix=s3_folder_name + '/'):
        # if the file already downloaded, just continue
        if f.key.split('/')[-1] in already_download_list:
            continue
        if f.key.endswith('.txt'):
            file = client.get_object(Bucket=bucket_name, Key=f.key)['Body'].read()    # write it to memory
            try:
                sftp = ssh.open_sftp()
                with sftp.open(os.path.join(sftp_path, f.key.split('/')[-1]), 'w') as f_w:
                    f_w.write(file)
            except Exception as e:
                # after the whole step finished, here should just save the rusult to DataFrame to local disk,
                # if with error, just save the already download files to disk, so that here the file is just
                # for the files that already downloaded
                save_downloaded(already_download_list)
                logger.warning('When download to SFTP with error: {}'.format(e))

            i += 1
            already_download_list.append(f.key.split('/')[-1])
        if i % 10000 == 0:
            print('Already downloaded {} files '.format(i))
            logging.info('Now have downloading {} files'.format(i))

    logging.info('Whole step finished!')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('is_first_time', type=bool, default=True)

    args = parser.parse_args()

    # first is to init the ssh for remote connection
    ssh = init_sftp()

    download_to_sftp(ssh, False)

