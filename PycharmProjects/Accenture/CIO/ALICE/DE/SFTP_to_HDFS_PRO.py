# -*- coding:utf-8 -*-
import paramiko
from paramiko import AuthenticationException, SSHException
import configparser
import os
from inspect import getsourcefile
import boto3
from boto3.exceptions import ResourceLoadException

# This function is used to execute the given command
def execute_command(ssh, command):
    print('Now is %s'%(command))
    stdin, stdout, stderr = ssh.exec_command(command)
    return stdin, stdout, stderr


# This is the main function to execute the logic to do file transfer
def file_transfer(config):
    host = config['config']['host']
    port = int(config['config']['port'])
    username = config['config']['username']
    password = config['config']['password']

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(host, port=port, username=username, password=password, look_for_keys=False)
        # transport = ssh.get_transport()
        # transport.auth_none(username)

        # Here I have to make date folder for SFTP and HDFS, if not given then just get the current date
        import datetime

        curr_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y%m%d')
        # sftp_path = "/sftp/cio.alice"
        # hdfs_path = "/data/raw/cio/alice"

        multi_command = """
            mkdir -p /sftp/cio.alice/%s &&
            mv /sftp/cio.alice/*.txt /sftp/cio.alice/%s/ &&
            hdfs dfs -mkdir -p /data/insight/cio/alice/contracts_files/%s &&
            hdfs dfs -put -f /sftp/cio.alice/%s/*.txt /data/insight/cio/alice/contracts_files/%s/  &&
            hdfs dfs -put -f /sftp/cio.alice/%s/*.txt /data/insight/cio/alice/contracts_files/whole_files2/
            """ % (curr_date, curr_date, curr_date, curr_date, curr_date, curr_date)

        # # This is demo step
        # multi_command = """
        #             mkdir -p /sftp/cio.alice/%s &&
        #             mv /sftp/cio.alice/*.txt /sftp/cio.alice/%s/ &&
        #             hdfs dfs -mkdir -p /data/insight/cio/alice/demo/contracts_files/%s &&
        #             hdfs dfs -put -f /sftp/cio.alice/%s/*.txt /data/insight/cio/alice/demo/contracts_files/%s/  &&
        #             hdfs dfs -put -f /sftp/cio.alice/%s/*.txt /data/insight/cio/alice/demo/contracts_files/whole_files/
        #             """ % (curr_date, curr_date, curr_date, curr_date, curr_date, curr_date)

        print('Starting executing: ')

        # multi_command = "cp /sftp/cio.alice/newer/*.txt /sftp/cio.alice/older && hdfs dfs -put -f /sftp/cio.alice/newer/*.txt /data/raw/cio/alice/test/newer/ "
        execute_command(ssh, multi_command)

        # show_hdfs_files('%s/newer'%(hdfs_path))

    except (AuthenticationException, SSHException) as e:
        print('when connect to SSH, with error: %s' % (e))


# This function is used to copy one folder files to another folder, after copy step finished,
# then just remove the whole files in the source folder
def move_folder_to_folder_for_s3(config):
    access_key = config['config']['access_key']
    secret_key = config['config']['secret_key']
    bucket_name = config['config']['bucket_name']
    src_folder_name = config['config']['src_folder_name']
    des_folder_name = config['config']['des_folder_name']

    try:
        s3 = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key).resource('s3')
        my_bucket = s3.Bucket(bucket_name)

        # first to get whole files in s3 bucket
        file_list = []
        for f in my_bucket.objects.filter(Prefix=src_folder_name + '/'):
            if f.key.endswith('.txt'):
                file_list.append(f.key)

        # ['/Detla/a.txt'] - > ['/TextFiles/a.txt']
        # use this code to copy files from one folder to another folder, change the desc file prefix
        des_file_list = [x.replace(src_folder_name, des_folder_name) for x in file_list]

        # 2 steps: 1. copy files from source folder ->desc folder
        for i in range(len(file_list)):
            src_dirc = {'Bucket': bucket_name, 'Key': file_list[i]}
            s3.meta.client.copy(src_dirc, bucket_name, des_file_list[i])

        # Then here is to remove the files in the source part, for now shouldn't used as if here delete the files,
        for f in file_list:
            s3.Object(bucket_name, f).delete()

    except ResourceLoadException as e:
        print('Connect to S3 with failure: %s'%(e))

    print('S3 step has finished!')


if __name__ == '__main__':
    # This is to get the config content
    config_path = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))
    config_name = 'config.conf'

    config = configparser.ConfigParser()
    config.read(os.path.join(config_path, config_name))

    # This is the main logic to do file transfer
    file_transfer(config)

    # After SFTP step finished, the trigger the copy folder from one folder to another folder for S3 bucket
    move_folder_to_folder_for_s3(config)

    print('Now All step run successfully without error!')