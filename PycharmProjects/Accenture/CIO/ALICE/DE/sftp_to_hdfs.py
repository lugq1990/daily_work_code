# -*- coding:utf-8 -*-
import paramiko


def execute_command(ssh, command_str):
    try:
       print('Now is : %s' % (command_str))
       stdin, stdout, stderr = ssh.exec_command(command_str)
       stdout.readlines()
    except RuntimeError:
        raise RuntimeError('Run error for command: {}'.format(command_str))
    return stdin, stdout, stderr



# This function is used to connect to SFTP, move files from 'newer' folder to 'older' folder
# This is what steps that needs to be processed

# Adjust for step: prior step should move file from HDFS newer to older, because I have to store
# the newest added file to be processed in this day, tomorrow I will first move file to older, then other step,
# so that newer added file will be stored for one day to be processed for bellow step!

# 1. Copy newer folder files to older;
# 2. clear the newer folder all files;
# 3. Check if whether or not the file name is already exits, if exits, just overwrite
def sftp_transfer(host, username, password, sftp_path, hdfs_path, hdfs_handler=None, remover_newer_folder_files=False):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=22, username=username, password=password)


    # This is command for remove all files in newer folder in HDFS
    remove_newer_hdfs = "hdfs dfs -rm %s/newer/*.txt" % (hdfs_path)

    # This is just one example that uses command to copy files
    copy_command = "cp %s/newer/*.txt %s/older/" % (sftp_path, sftp_path)
    # This command is used to clear the 'newer' folder, remove all files
    remove_newer_sftp = "rm * %s/newer//" % (sftp_path)
    # This command is used to put sftp files to HDFS
    file_sftp_to_hdfs = "hdfs dfs -put %s/newer/*.txt %s/newer/" % (sftp_path, hdfs_path)

    # This command is used to move files from 'newer' folder to 'older' folder
    hdfs_newer_to_older = "hdfs dfs -mv %s/newer/*.txt %s/older/" % (hdfs_path, hdfs_path)


    execute_command(ssh, remove_newer_hdfs)
    # This should be 1.copy from sftp newer folder to older, 2.convert file from sftp to HDFS,
    # 3.copy files from HDFS newer to older
    execute_command(ssh, copy_command)
    execute_command(ssh, file_sftp_to_hdfs)

    # Because by using command can't move HDFS file folder from newer to older,
    # So here just use hdfs_handler to process it.

    if hdfs_handler is not None:
        print('Now is just using hdfs_handler to move files!')
        hdfs_handler.mv('%s/newer/*.txt' % hdfs_path, '%s/older/' % hdfs_path)
    else:
        execute_command(ssh, hdfs_newer_to_older)


    # Before I run the copy command, first just remove newer folder in HDFS(because for now, HDFS can't be removed),
    # so that files can be stored for one day
    # execute_command(ssh, file_sftp_to_hdfs)

    if remover_newer_folder_files:
        execute_command(ssh, remove_newer_sftp)

    return True
    # transport = paramiko.Transport((host, port))
    # sftp = paramiko.SFTPClient.from_transport(transport)
    # sftp.listdir(path)
    # sftp.close()
    # transport.close()


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
    sftp_path = '/sftp/cio.alice'
    hdfs_path = '/data/raw/cio/alice/test'

    port = 22
    # username = 'xueting.zhu'
    # password = 'XXXttt@2019'
    username = 'guangqiang.lu'
    password = 'Lugq1990!'

    from hdfs3 import HDFileSystem
    # This is for HDFS config
    host = 'name-node.cioprd.local'
    conf = {'dfs.nameservices': 'cioprdha', 'hadoop.security.authentication': 'kerberos'}
    hdfs_handler = HDFileSystem(pars=conf, host=host, port=8020)

    # host = '10.5.105.51'
    # port = 22
    # # username = 'ngap.app.alice'
    # # password = 'QWer@#2019'
    # username = 'guangqiang.lu'
    # password = 'Lugq1990!'
    # path = '/sftp/cio.alice'
    # hdfs_path = '/tmp/sftp/newer/tmp/20190226/code'


    debug = True

    # Start to process sftp files transfer
    sftp_transfer(host, username, password, sftp_path, hdfs_path)

    # if args.show_hdfs is not None or debug:
    if debug:
        show_hdfs_files(hdfs_handler, hdfs_path, True)


