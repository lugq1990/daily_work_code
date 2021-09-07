# -*- coding:utf-8 -*-
import paramiko
from paramiko import AuthenticationException, SSHException

# This function is used to execute the given command
def execute_command(ssh, command):
    print('Now is %s'%(command))
    stdin, stdout, stderr = ssh.exec_command(command)
    return stdin, stdout, stderr

# This function is used to show HDFS folders files
def show_hdfs_files(hdfs_path, show_in_terminal=True, n_show=10):
    from hdfs3 import HDFileSystem
    # This is for HDFS config
    host = 'name-node.cioprd.local'
    conf = {'dfs.nameservices': 'cioprdha', 'hadoop.security.authentication': 'kerberos'}
    hdfs_handler = HDFileSystem(pars=conf, host=host, port=8020)

    files = hdfs_handler.ls(hdfs_path)

    # Loop for all result
    if show_in_terminal:
        print('Showing %s HDFS folder top %d files'%(hdfs_path, n_show))
        for i in range(len(files)):
            print('This is %dth file, name: %s'%(i, files[i]['name'].split('/')[-1]))
            if i > n_show:
                break


if __name__ == '__main__':
    host = '10.5.105.51'
    port = 22
    username = 'ngap.app.alice'
    password = 'QWer@#2019'
    sftp_path = '/sftp/cio.alice'
    hdfs_path = '/data/raw/cio/alice/test'

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
        ### this is before
        # multi_command = """
        # mkdir -p /sftp/cio.alice/%s && cp /sftp/cio.alice/newer/*.txt /sftp/cio.alice/%s/ &&
        # hdfs dfs -mkdir -p /data/raw/cio/alice/%s && hdfs dfs -put -f /sftp/cio.alice/newer/*.txt /data/raw/cio/alice/%s/
        # """%(curr_date, curr_date, curr_date, curr_date)
        multi_command = """
        mkdir -p /sftp/cio.alice/%s && 
        mv /sftp/cio.alice/*.txt /sftp/cio.alice/%s/ && 
        hdfs dfs -mkdir -p /data/raw/cio/alice/%s && 
        hdfs dfs -put -f /sftp/cio.alice/%s/*.txt /data/raw/cio/alice/%s/ 
        """ % (curr_date, curr_date, curr_date, curr_date, curr_date)

        print('Starting executing: ')

        # multi_command = "cp /sftp/cio.alice/newer/*.txt /sftp/cio.alice/older && hdfs dfs -put -f /sftp/cio.alice/newer/*.txt /data/raw/cio/alice/test/newer/ "
        execute_command(ssh, multi_command)

        # show_hdfs_files('%s/newer'%(hdfs_path))

        print('Now All step run successfully without error!')

    except (AuthenticationException, SSHException) as e:
        print('when connect to SSH, with error: %s'%(e))

    # # Becasue I don't want to run bellow command right now, so here just wait some time
    # # print('Now code will wait for 10 seconds')
    # # import time
    # # time.sleep(10)
    #
    # # Here I have to make date folder for SFTP and HDFS, if not given then just get the current date
    # import datetime
    # curr_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y%m%d')
    # # sftp_path = "/sftp/cio.alice"
    # # hdfs_path = "/data/raw/cio/alice"
    # ### this is before
    # # multi_command = """
    # # mkdir -p /sftp/cio.alice/%s && cp /sftp/cio.alice/newer/*.txt /sftp/cio.alice/%s/ &&
    # # hdfs dfs -mkdir -p /data/raw/cio/alice/%s && hdfs dfs -put -f /sftp/cio.alice/newer/*.txt /data/raw/cio/alice/%s/
    # # """%(curr_date, curr_date, curr_date, curr_date)
    # multi_command = """
    # mkdir -p /sftp/cio.alice/%s &&
    # cp /sftp/cio.alice/contract_tmp/*.txt /sftp/cio.alice/%s/ &&
    # hdfs dfs -mkdir -p /data/raw/cio/alice/%s &&
    # hdfs dfs -put -f /sftp/cio.alice/%s/*.txt /data/raw/cio/alice/%s/
    # """%(curr_date, curr_date, curr_date, curr_date, curr_date)
    #
    # print('Starting executing: ')
    #
    # # multi_command = "cp /sftp/cio.alice/newer/*.txt /sftp/cio.alice/older && hdfs dfs -put -f /sftp/cio.alice/newer/*.txt /data/raw/cio/alice/test/newer/ "
    # execute_command(ssh, multi_command)
    #
    # # show_hdfs_files('%s/newer'%(hdfs_path))
    #
    # print('Now All step run successfully without error!')




# just to get whole file size that I have trained model based on the MML server.
import os

path = os.curdir
folder_list = [x for x in os.listdir('.') if os.path.isdir(x)]

file_size_dict = {}
# loop for each folder
for f in folder_list:
    file_list = os.listdir(f)
    for file in file_list:
        file_size_dict[file] = os.path.getsize(os.path.join(path, f, file))

file_size = [(file, os.path.getsize(os.path.join(path, f, file)))
             for f in folder_list for file in os.listdir(f) if not file.endswith('csv') and 'pre' not in file]

file_size_dict = {}
for file, size in file_size:
    if int(size / 1024**2) >= 1:
        file_size_dict[file] = str(round(size / 1024**2, 2)) + ' MB'
    elif int(size / 1024) > 1 and int(size / 1024**2) < 1:
        file_size_dict[file] = str(round(size / 1024, 2)) + ' KB'
    else:
        file_size_dict[file] = str(size) + ' Bytes'
