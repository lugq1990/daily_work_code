import paramiko
import schedule
import time


ip = "10.5.105.51"
user_name = "ngap.app.dsd"
password = r"Aview#4%aoR7"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(ip, username=user_name, password=password)


def monitor_ssh():
    global ssh
    # in case that SSH client will be in-activate, just create a new SSH client
    print("Start to check SSH client")
    if not ssh.get_transport().active:
        ssh = create_ssh()
        print("New SSH client has been created!")


def create_ssh():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, username=user_name, password=password)
    
    return ssh


def shell_job_phi():
    command = r"""nohup sh /home/ngap.app.dsd/aia_uccx_migration/backup_ebi.sh philippines > /home/ngap.app.dsd/aia_uccx_migration/philippines/BackupLog-`date +%y%m%d`-`date +%H%M`.txt 2>&1 &"""
    stdin, stdout, stderr = ssh.exec_command(command)
    print(stdout.read())


def shell_job_brazil():
    command = r"""nohup sh /home/ngap.app.dsd/aia_uccx_migration/backup_ebi.sh brazil > /home/ngap.app.dsd/aia_uccx_migration/brazil/BackupLog-`date +%y%m%d`-`date +%H%M`.txt 2>&1 &"""
    stdin, stdout, stderr = ssh.exec_command(command)
    print(stdout.read())


def shell_job_india():
    command = r"""nohup sh /home/ngap.app.dsd/aia_uccx_migration/backup_ebi.sh india > /home/ngap.app.dsd/aia_uccx_migration/india/BackupLog-`date +%y%m%d`-`date +%H%M`.txt 2>&1 &"""
    stdin, stdout, stderr = ssh.exec_command(command)
    print(stdout.read())


if __name__ == "__main__":
    schedule.every().day.at("19:48").do(shell_job_phi)
    schedule.every().day.at("22:35").do(shell_job_brazil)
    schedule.every().day.at("14:50").do(shell_job_india)

    # With every 10 mins to check SSH status to make it workable.
    # schedule.every(1).minutes.do(disable_ssh)
    schedule.every(10).minutes.do(monitor_ssh)
    
    while True:
        schedule.run_pending()

