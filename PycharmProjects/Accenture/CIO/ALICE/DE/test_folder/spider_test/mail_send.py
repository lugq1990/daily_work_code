# -*- coding:utf-8 -*-
"""This is to test to send mail from the server."""
import smtplib
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart

# sender = 'gqianglu1990@gmail.com'
sender = 'guangqiang.lu@accenture.com'
receiver = ['guangqiang.lu@accenture.com']

# message = """From: Lugq <gqianglu@outlook.com>
# To: Lugq <guangqiang.lu@accenture.com>
# Subject: Test for python
#
# This is just to test python send mail."""
# message = EmailMessage()
# message['Subject'] = 'python email'
# message['From'] = sender
# message['To'] = receiver[0]

message = EmailMessage()
message.add_header('from', sender)
message.add_header('to', receiver[0])
message.add_header('subject', 'Jest to test python module')
message.set_payload('test\n for machine learning for later use case')

message = MIMEMultipart()
message['From'] = sender
message['To'] = receiver[0]
message['Subject'] = 'machine learning'




smtp_host = 'smtp.office365.com'
# smtp_port = 587
# smtp_host = 'smtp-mail.outlook.com'
smtp_port = 587
username = 'gqianglu@outlook.com'
password = '1131298218a'

smtp_host = 'smtp.gmail.com'
smtp_port = 587
username = 'gqianglu1990@gmail.com'
password = '1131298218a'

# so here just try to use localhost as SMTP server to send mail
try:
    smtp = smtplib.SMTP('localhost', timeout=120)
    smtp.set_debuglevel(1)
    smtp.ehlo()
    res = smtp.sendmail(sender, receiver, message.as_string())
    print("Send mail successfully! with res dictionary: %s" % str(res))
except Exception as e:
    print("Sent mail with error: %s" % e)



# as for my local server, there is error with 10060 timeout error
# for using MML server, this doesn't support for outside port.

# try:
#     print('start to instant the SMTP object.')
#     # smtp = smtplib.SMTP(smtp_host, smtp_port, timeout=120)
#     # smtp = smtplib.SMTP(host='localhost', port=1025)
#     smtp = smtplib.SMTP_SSL(smtp_host, 465)
#     smtp.set_debuglevel(1)
#     smtp.ehlo()
#     print('init smtp')
#     smtp.login(username, password)
#     print('login successfully.')
#     smtp.sendmail(sender, receiver, message)
#     print("Send mail successfully!")
# except Exception as e:
#     print("Send mail fail as %s" % e)



# here test with yagmail
# import yagmail
#
# yagmail.register(username, password)
# yag = yagmail.SMTP('smtp.gmail.com:465', timeout=120)
# yag.send(to=receiver, subject='Test for python', contents='this is body for testing')
# print('Sent!')




# import smtplib
#
# FROMADDR = username
# LOGIN    = FROMADDR
# PASSWORD = password
# TOADDRS  = receiver
# SUBJECT  = "Test"
#
# msg = "From: %s\r\n To: %s\r\n Subject: %s\r\n\r\n" % (FROMADDR, ", ".join(TOADDRS), SUBJECT)
# msg += "some text\r\n"
#
# server = smtplib.SMTP('smtp.gmail.com', 465)
# server.set_debuglevel(1)
# server.ehlo()
# server.starttls()
# server.login(LOGIN, PASSWORD)
# server.sendmail(FROMADDR, TOADDRS, msg)
# server.quit()


import smtplib
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import shutil
import tempfile

# sender = 'gqianglu1990@gmail.com'
sender = 'guangqiang.lu@accenture.com'
receiver = ['1131298218@qq.com']

# this just want to use the reading files content solution to do this
tmp_path = tempfile.mkdtemp()
file_path = os.path.join(tmp_path, 't.txt')
with open(file_path, 'w') as fw:
    fw.write("This is a machine learning test for python email function")

with open(file_path, 'r') as fr:
    msg = MIMEText(fr.read())

msg['Subject'] = 'the content of %s' % file_path
msg['From'] = sender
msg['To'] = receiver[0]

server = smtplib.SMTP('localhost')
server.sendmail(sender, [receiver], msg.as_string())


# message = EmailMessage()
# message.add_header('from', sender)
# message.add_header('to', receiver[0])
# message.add_header('subject', 'Jest to test python module')
# message.set_payload('test\n for machine learning for later use case')

smtp_host = 'smtp.gmail.com'
smtp_port = 587
username = 'gqianglu1990@gmail.com'
password = '1131298218a'

import ssl
context = ssl.create_default_context()

try:
    server = smtplib.SMTP(smtp_host, smtp_port)
    server.ehlo()
    server.starttls(context)
    server.login(username, password)
except Exception as e:
    print("When login with error: %s" % e)
finally:
    server.quit()



"""This is just to make the mapping file to one date"""
import os
import shutil

cur_path = os.path.abspath(os.curdir)
des_folder = 'fix_all'


folder_list = [x for x in os.listdir(cur_path) if x.endswith('fix')]

folder_name = folder_list[1]

mapping_file_list = [x for x in os.listdir(os.path.join(cur_path, folder_name)) if x.lower().startswith('mapping')]
data = []
for f in mapping_file_list:
    with open('/'.join([cur_path, folder_name, f]), 'r') as fr:
        data.extend(fr.readlines())

# change mapping file name
new_single_mapping_file_name = 'MappingFile_20190104_0001.txt'
# store the new mapping file to des_folder
with open('/'.join([cur_path, des_folder, new_single_mapping_file_name]), 'w') as fw:
    fw.write(''.join(data))

# copy files from source folder to destination folder
file_list = [x for x in os.listdir(os.path.join(cur_path, folder_name)) if not x.lower().startswith('mapping')]
for f in file_list:
    shutil.copy('/'.join([cur_path, folder_name, f]), '/'.join([cur_path, des_folder, f]))
    if file_list.index(f) % 5000 == 0:
        print("Already copied %d files." % file_list.index(f))


des_mapping = [x for x in os.listdir(os.path.join(cur_path, des_folder)) if x.lower().startswith('mapping')]
print("There exists: %s" % str(des_mapping))

print('there are %d files in destination folder!' % len(os.listdir(os.path.join(cur_path, des_folder))))

desc_files = [x for x in os.listdir(os.path.join(cur_path, des_folder)) if not os.path.isdir('/'.join([cur_path, des_folder, x]))]
for f in desc_files:
    os.remove('/'.join([cur_path, des_folder, f]))
    if desc_files.index(f) % 5000 == 0:
        print("already removed %d files." % desc_files.index(f))


"""Here is to check whether there are some files same for new added and whole files"""
from hdfs.ext.kerberos import KerberosClient
import os
import pandas as pd
import tempfile

new_hdfs_path = '/data/insight/cio/alice.pp/hivetable/documents_name'
whole_hdfs_path = '/data/insight/cio/alice/hivetable/documents_name'

client = KerberosClient("http://name-node.cioprd.local:50070")

new_files_list = client.list(new_hdfs_path)
whole_files_list = client.list(whole_hdfs_path)

new_list = []
whole_list = []

# here download files to temperate folder and read it with pandas
tmp_path = tempfile.mkdtemp()
new_df = pd.DataFrame()
whole_df = pd.DataFrame()

files_tuple = (new_files_list, whole_files_list)

for folder_list in files_tuple:
    for f in folder_list:
        print("Now is %s file." % f)
        if files_tuple.index(folder_list) == 0:
            client.download(hdfs_path=os.path.join(new_hdfs_path, f), local_path=os.path.join(tmp_path, f),
                            overwrite=True)
            tmp_df = pd.read_csv(os.path.join(tmp_path, f))
            new_df = pd.concat((new_df, tmp_df), axis=0)
        else:
            client.download(hdfs_path=os.path.join(whole_hdfs_path, f), local_path=os.path.join(tmp_path, f),
                            overwrite=True)
            tmp_df = pd.read_csv(os.path.join(tmp_path, f))
            whole_df = pd.concat((whole_df, tmp_df), axis=0)



