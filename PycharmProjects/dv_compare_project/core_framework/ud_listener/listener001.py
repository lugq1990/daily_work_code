#!/usr/bin/env python
import os

import config
import devops_api_utils
import robot_auto_test
import threading
import time
import pymsteams

ROBOT_LISTENER_API_VERSION = 2


class MyThread(threading.Thread):
    def __init__(self, n):
        super(MyThread, self).__init__()  # 重构run函数必须要写
        self.n = n

    def run(self):
        print("task", self.n)
        time.sleep(1)
        print('2s')
        time.sleep(1)
        print('1s')
        time.sleep(1)
        print('0s')
        time.sleep(1)


MyThread("t1").start()


def start_suite(name, attrs):
    print('PythonListener---- ' + 'start_suite name:' + name)
    # print(attrs)
    if name.startswith('Test Case'):
        # xunit_file = get_latest_xunit(os.path.join(robot_auto_test.output_dir, 'xunit_files'))
        xunit_file = os.path.join(robot_auto_test.output_dir, 'xunit_files', 'xunit_dryrun.xml')
        devops_api_utils.create_test_run_by_xunit(xunit_file)


def start_test(name, attrs):
    # print('PythonListener---- ' + 'start_test: %s' % attrs)

    ado_update_test_result_lst = []
    test_point_id = devops_api_utils.TEST_POINT_CASE_DICT[name]
    result_id = devops_api_utils.TEST_POINT_RESULT_DICT[test_point_id]
    ado_update_test_result_lst.append({"id": str(result_id),
                                       # "duration_in_ms": 0,
                                       "outcome": 'InProgress',
                                       "error_message": ''})
    devops_api_utils.update_test_results(ado_update_test_result_lst)


def end_test(name, attrs):
    # print('PythonListener---- ' + 'end_test: %s' % attrs)

    ado_update_test_result_lst = []
    test_point_id = devops_api_utils.TEST_POINT_CASE_DICT[name]
    result_id = devops_api_utils.TEST_POINT_RESULT_DICT[test_point_id]

    outcome = 'Passed' if attrs['status'] == 'PASS' else 'Failed'
    ado_update_test_result_lst.append({"id": str(result_id),
                                       "duration_in_ms": attrs['elapsedtime'],
                                       "outcome": outcome,
                                       "error_message": attrs['message'],
                                       "state": "Completed"})
    devops_api_utils.update_test_results(ado_update_test_result_lst)


def end_suite(name, attrs):
    print('PythonListener---- ' + 'end_suite')


def close():
    devops_api_utils.finalize_test_run(os.path.join(robot_auto_test.output_dir, 'log.html'))
    webhook = 'https://accenture.webhook.office.com/webhookb2/66b0efab-7b4d-432c-a935-ae3fb15ff262@e0793d39-0939-496d-b129-198edd916feb/IncomingWebhook/19363efe273a4b9f9e691e766f05ab54/003a2cfe-c9cc-43fb-b7cf-46b3238a483f'
    myTeamsMessage = pymsteams.connectorcard(webhook)

    link = "{}/{}/_testManagement/runs?runId={}&_a=resultQuery".format(config.organization_url, config.project_id, devops_api_utils.TEST_RUN_ID)
    myTeamsMessage.text('This is an notification email for the completion of your test plan.<br/> Please Click this <a href=\'' + link + '\'>link</a> to review the test results.!')
    myTeamsMessage.send()

    print('PythonListener---- ' + 'close')


def xunit_file1(path):
    with open(path, 'r') as f:
        print(f.readlines())


def get_latest_xunit(path):
    lists = os.listdir(path)
    lists.sort(key=lambda fn: os.path.getmtime(os.path.join(path, fn)))
    file_new = os.path.join(path, lists[-1])
    return file_new


if __name__ == "__main__":
    test_report = r"C:\projects\code\Alice\dv-auto-test\core_framework\output\xunit_files"
    # print(get_latest_xunit(test_report))
    close()
