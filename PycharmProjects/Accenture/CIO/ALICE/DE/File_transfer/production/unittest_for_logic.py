# -*- coding:utf-8 -*-

import paramiko
from paramiko import AuthenticationException
import time
import sys
import unittest

retry_times = 3

host = "10.5.105.51"
port = 22
username = "ngap.app.alice"
password = "QWer@#2019"

config = {"config":{'host':host, 'port': port, 'username': username, 'password': password}}


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

    # if after many time to init with error, just raise one error to the system
    if ssh is None or sftp is None:
        sys.exit("couldn't init the SSH object for %d times" % (retry_times))

    return ssh, sftp


class TestInit(unittest.TestCase):

    def test_nan(self):
        self.assertIsNotNone(init_ssh(config))

    def test_equal(self):
        # first should init manually, then compare with the current
        # object then compare with manually instance object and function return object
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        ssh.connect(host, port, username, password, look_for_keys=False)
        sftp = ssh.open_sftp()

        ssh_func, sftp_func = init_ssh(config)

        self.assertEqual(ssh.__class__, ssh_func.__class__)
        self.assertEqual(sftp.__class__, sftp_func.__class__)


if __name__ == '__main__':
    unittest.main()