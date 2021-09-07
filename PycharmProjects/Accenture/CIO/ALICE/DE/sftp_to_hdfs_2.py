# -*- coding:utf-8 -*-
import atexit
import paramiko

class MySSH:
    def __init__(self, host, username, password, port = 22):
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host, port=port, username=username, password=password)
        # atexit.register(client.close)
        self.client = client

    def __call__(self, command):
        print('Now is command:', command)
        stdin, stdout, stderr = self.client.exec_command(command)
        # sshdata = stdout.readlines()
        # for line in sshdata:
        #     print(line)


# After load file to HDFS, here I use hdfs3 to read HDFS files as a list result
def show_hdfs_files(hdfs_handler, hdfs_path, show_result_in_terminal=True, show_numbers=20):
    hdfs_result_folder = '%s/older/'%(hdfs_path)
    result = hdfs_handler.ls(hdfs_result_folder)

    # Loop for the result to print first 10 result to terminal
    if show_result_in_terminal:
        print('There are %d files in HDFS older folder'%(len(result)))
        for i in range(len(result)):
            print('This is {0:2d}th file, file name: {1} in HDFS folder.'
                  .format(i, result[i]['name'].split('/')[-1]))
            if i > show_numbers: break

    return result


if __name__ == '__main__':
    host = '10.5.105.51'
    port = 22
    username = 'xueting.zhu'
    password = 'XXXttt@2019'
    sftp_path = '/sftp/cio.alice'
    hdfs_path = '/data/raw/cio/alice/test'

    from hdfs3 import HDFileSystem
    # This is for HDFS config
    host = 'name-node.cioprd.local'
    conf = {'dfs.nameservices': 'cioprdha', 'hadoop.security.authentication': 'kerberos'}
    hdfs_handler = HDFileSystem(pars=conf, host=host, port=8020)


    # This is command for remove all files in newer folder in HDFS
    remove_newer_hdfs = "hdfs dfs -rm %s/newer/*.txt" % (hdfs_path)

    # This is just one example that uses command to copy files
    copy_command = "cp %s/newer/*.txt %s/older/" % (sftp_path, sftp_path)
    # This command is used to clear the 'newer' folder, remove all files
    remove_newer_sftp = "rm * %s/newer/" % (sftp_path)
    # This command is used to put sftp files to HDFS
    file_sftp_to_hdfs = "hdfs dfs -put %s/newer/*.txt %s/newer/" % (sftp_path, hdfs_path)

    # This command is used to move files from 'newer' folder to 'older' folder
    hdfs_newer_to_older = "hdfs dfs -mv %s/newer/*.txt %s/older/" % (hdfs_path, hdfs_path)

    execute_command = MySSH(host, username, password, port)
    """Here because I want to execute multiple command same time, here by google answer"""
    multi_c = "cp /sftp/cio.alice/newer/*.txt /sftp/cio.alice/older/ && cd /sftp/cio.alice && ls"
    multi_c = 'cp /sftp/cio.alice/newer/*.txt /sftp/cio.alice/older/'
    execute_command(multi_c)


    # Here start to execute each command
    # execute_command(remove_newer_hdfs)
    # execute_command(copy_command)

    # Here I maybe just not to want to use paramiko to do File transfer for HDFS
    # print('Here I just use hdfs3 to put file and move files')

    # Put command
    # hdfs_handler.put('%s/newer/*.txt'%sftp_path, '%s/newer'%hdfs_path)
    # Copy command
    # hdfs_handler.mv('%s/newer/*.txt'%hdfs_path, '%s/older/'%hdfs_path)

    #
    # execute_command(file_sftp_to_hdfs)
    # execute_command(hdfs_newer_to_older)
    #
    # show_hdfs_files(hdfs_handler, hdfs_path)
    # print('All finished!')


    # Because for just using python code to execute command will meet thread problem, here I will just run the
    # predefined .sh file to execute command!
    # ssh_file_path = '/sftp/cio.alice/newer'
    # ssh_command = 'sh %s/t.sh'%ssh_file_path
    #
    # execute_command('chmod 777 %s/t.sh'%ssh_file_path)
    # execute_command(ssh_command)
    #
    # show_hdfs_files(hdfs_handler, hdfs_path)
    # print('All finished!')

