# -*- coding:utf-8 -*-
"""This is used to do mapping file problem to put the mapping file to the external HDFS path for later query,
I also add a function to do the download S3 bucket files to the MML server, and use scp to copy this files to SFTP,
after that I will remove the whole files in the MML folder. But for now, maybe not used."""
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

### This function is used to do file downlaod to MML and put files to SFTP and remove the downloaded files
# this will make a temperate folder and after use it, just remove it
def download_file_from_s3_to_sftp():
    import boto3
    import shutil
    import tempfile

    # TODO: if this function to be used for production, here will use the config file
    # for now, just explicate writen
    access_key = 'AKIAJGK5GH7CU46OSS5Q'
    secret_key = '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'
    bucket_name = 'aliceportal-30899-prod'
    s3_folder_name = 'Delta'
    s3 = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key).resource('s3')
    my_bucket = s3.Bucket(bucket_name)

    # here is just to make a temperate folder
    tmp_folder_in_mml = tempfile.mkdtemp()

    # downlaod files to temperate folder
    for f in my_bucket.objects.filter(Prefix=s3_folder_name + '/'):
        if f.key.endswith('.txt'):
            my_bucket.Object(f).download_file(tmp_folder_in_mml, f.key.split('/')[-1])

    # After download files step, use scp to copy files to production SFTP folder
    copy_commnad = "scp -r {}/*.txt ngap.app.alice@10.5.105.51:/sftp/cio.alice".format(tmp_folder_in_mml)

    # after the whole process finish, remove the temperate folder
    try:
        [os.system(copy_commnad)]
        shutil.rmtree(tmp_folder_in_mml)
    except Exception as e:
        pass


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


        # this is added mapping file with ***.txt files to another HDFS path where the external table created
        external_path = 'hdfs://cioprdha/data/insight/cio/alice/external/filepath_mapping'
        mapping_file_name = 'mapping.txt'
        sftp_path = '/sftp/cio.alice'
        mapping_file_path = os.path.join(sftp_path, mapping_file_name)

        multi_command = """
            hdfs dfs -put -f {} {} &&
            rm -f {} &&
        """.format(mapping_file_path, external_path, mapping_file_path)


        # This is combined with previous step
        multi_command += """
            mkdir -p /sftp/cio.alice/%s && 
            mv /sftp/cio.alice/*.txt /sftp/cio.alice/%s/ && 
            hdfs dfs -mkdir -p /data/insight/cio/alice/contracts_files/%s && 
            hdfs dfs -put -f /sftp/cio.alice/%s/*.txt /data/insight/cio/alice/contracts_files/%s/ &&
            hdfs dfs -put -f /sftp/cio.alice/%s/*.txt /data/insight/cio/alice/contracts_files/whole_files2/
            """ % (curr_date, curr_date, curr_date, curr_date, curr_date, curr_date)

        print('Starting executing: ')

        execute_command(ssh, multi_command)

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
        # use this code to copy files from one folder to another folder, change the desc file prefix
        des_file_list = [x.replace(src_folder_name, des_folder_name) for x in file_list]

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

