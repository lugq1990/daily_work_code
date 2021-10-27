#!/usr/bin/env python
import json
from robot.api.deco import keyword, library
from SeleniumLibrary import SeleniumLibrary
from selenium.webdriver import ActionChains
from selenium.webdriver import ChromeOptions
import time
import pandas as pd
from robot.libraries.BuiltIn import BuiltIn
import platform
from google.api_core import retry
from google.cloud import pubsub_v1
import os
import config
import shutil

sa = r'C:\Users\jianglei.chen\OneDrive - Accenture\Documents\Downloads\sbx-65343-autotest12--260babe0-2d6f5943c330.json'


@library
class SeleniumLibraryExtension(SeleniumLibrary):
    ROBOT_LIBRARY_SCOPE = 'GLOBAL'
    __version__ = '0.1'
    downloads_path = None

    @keyword("Open New Tab on Browser")
    def open_new_tab(self):
        self.driver.execute_script("window.open('');")
        window_list = self.driver.window_handles
        self.driver.switch_to.window(window_list[-1])

    @keyword('Set Cookie')
    def set_cookie(self, name, value, domain, path='/'):
        driver = self.driver
        driver.add_cookie({'name': name, 'value': value, 'domain': domain, 'path': path})

    @keyword("Right Click Element")
    def right_click(self, xpath):
        driver = self.driver
        results_path = BuiltIn().get_variable_value("${query}")
        print(results_path)
        action_chains = ActionChains(driver)
        if xpath.startswith('class:'):
            element = driver.find_element_by_class_name(xpath.split(':')[1])
        else:
            element = driver.find_element_by_xpath(str(xpath))
        action_chains.context_click(element).perform()
        # html = driver.execute_script("return document.documentElement.outerHTML")
        # print(html)

    @keyword("SET CHROME DOWNLOAD FOLDER")
    def set_chrome_download_folder(self, downloads_path, options=None):
        self.downloads_path = downloads_path
        print(downloads_path)
        if options is None:
            options = ChromeOptions()
        prefs = {"download.default_directory": str(downloads_path), }
        options.add_experimental_option('prefs', prefs)
        print(options)
        return options

    @keyword("GET OPTIONS")
    def get_options(self, options=None):
        if options is None:
            options = ChromeOptions()
        prefs = {"download.default_directory": str(
            os.path.join(BuiltIn().get_variable_value("${outputdir}"), 'tmpdownloads')), }
        options.add_experimental_option('prefs', prefs)
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        return options

    @keyword("Check And Return Download Files")
    def check_and_return_download_files(self):
        return get_download_file(os.path.join(BuiltIn().get_variable_value("${outputdir}"), 'tmpdownloads'))

    @keyword("Compare with database data")
    def compare_data_with_lake(self, downloads_path=None, has_header=True):
        if downloads_path is None:
            downloads_path = self.downloads_path

        expected_data_query = BuiltIn().get_variable_value("${query}")
        data_schema = BuiltIn().get_variable_value("${schema}")
        textobject_value = BuiltIn().get_variable_value("${textobject_value}")
        print(textobject_value)
        if textobject_value is None:
            downloads = self.check_and_return_download_files()
            print(downloads)
            if not os.path.exists(downloads_path):
                os.makedirs(downloads_path, exist_ok=True)
            shutil.move(
                os.path.join(os.path.join(BuiltIn().get_variable_value("${outputdir}"), 'tmpdownloads'), downloads[0]),
                os.path.join(downloads_path, downloads[0]))
            if len(downloads) == 1:
                processing(downloads_path, downloads[0], data_schema, expected_data_query, has_header)
        else:
            columns = json.loads(data_schema).keys()
            seg = zip(columns, textobject_value)
            j = dict(seg)
            print(j)
            j = [j]
            print(j)
            compare_with_lake(downloads_path.split(os.path.sep)[-1], "compare", j, data_schema, expected_data_query)
        # os.remove(downloads_path)

    @keyword("Compare Raw Data With Insight Data")
    def compare_raw_data_with_insight_data(self):
        actual_data_query = BuiltIn().get_variable_value("${query2}")
        expected_data_query = BuiltIn().get_variable_value("${query}")
        test_case_id = str(int(time.time()))
        data = r'{"id":"%s", "query":"%s","data":%s, "query2":"%s","mode":"%s"}' % (
            test_case_id, expected_data_query, {}, actual_data_query, "compare")
        send_to_pubsub_de(data.encode('utf-8'))
        print(data)
        get_test_result_from_pubsub_de(test_case_id)

    @keyword("Get Executable Path")
    def get_executable_path(self):
        if (platform.system() == 'Windows'):
            return r'C:\projects\code\Alice\dv-auto-test\core_framework\webdriver\chromedriver.exe'
        else:
            return '/usr/local/bin/chromedriver'


def processing(downloads_path, file_name, data_schema, expected_data_query, has_header=True):
    header = 0 if has_header else None
    if file_name.endswith('.xlsx'):
        data_schema = ""
        r = pd.read_excel(os.path.join(downloads_path, file_name), dtype='str')
    elif file_name.endswith('.csv'):
        data_schema = data_schema.replace('\'', '\"')
        columns = json.loads(data_schema).keys()
        r = pd.read_csv(os.path.join(downloads_path, file_name), encoding='utf8', dtype='str', header=header,
                        delimiter=';', names=columns)
    # r.columns = [c.replace(' ', '_') for c in r.columns]
    # r = pd.read_csv(r'C:\Users\jianglei.chen\Downloads\86f9e2fcac73457daf14626417194bbb.csv', encoding='utf8', dtype='str', delimiter=';')
    j = r.to_json(orient="records")
    compare_with_lake(downloads_path.split(os.path.sep)[-1], "compare", j, data_schema, expected_data_query)


def compare_with_lake(test_case_id, mode, data, data_schema, expected_data_query):
    data = create_message(test_case_id, mode, data, data_schema, expected_data_query)
    send_to_pubsub(data)
    print(data)
    get_test_result_from_pubsub2(test_case_id)
    # push_test_result_to_git_repo(downloads_path.split(os.path.sep)[-1])


def get_download_file(dirname):
    downloads = []
    check_times = 0
    while check_times < 5:
        downloads = os.listdir(dirname)
        for d in downloads:
            if d.endswith('crdownload'):
                time.sleep(5)
                check_times = check_times + 1

        if len(downloads) > 0:
            break
        else:
            time.sleep(5)
            check_times = check_times + 1

    return downloads


def create_message(test_case_id, mode, data, data_schema, query):
    # data_schema = data_schema.replace('\'', '\"')
    data = "%s" % data
    # data = data.replace('\'', '\"')
    data = r'{"id":"%s", "query":"%s","data":%s, "schema":"%s","mode":"%s"}' % (
        test_case_id, query, data, data_schema, mode)
    # data = data.replace('244', '245')
    # Data must be a byte string
    return data.encode("utf-8")


def send_to_pubsub(data):
    project_id = config.gcp_project_id  # BuiltIn().get_variable_value("${gcp_pubsub_project_id}")
    topic_id = config.pubsub_topic_id  # 'App_65343_COMPARE-DATA-TOPIC'  # BuiltIn().get_variable_value("${gcp_pubsub_topic_id}")
    # endpoint = "https://my-test-project.appspot.com/push"

    if ('Windows' == platform.system()):
        publisher = pubsub_v1.PublisherClient.from_service_account_json(sa)
    else:
        publisher = pubsub_v1.PublisherClient()
    # The `topic_path` method creates a fully qualified identifier
    # in the form `projects/{project_id}/topics/{topic_id}`
    topic_path = publisher.topic_path(project_id, topic_id)
    # When you publish a message, the client returns a future.
    future = publisher.publish(topic_path, data)
    print(future.result())


# TODO: Refactor below function to compatible for both DV and DE
def send_to_pubsub_de(data):
    project_id = config.gcp_project_id  # BuiltIn().get_variable_value("${gcp_pubsub_project_id}")
    topic_id = config.pubsub_topic_id_de  # 'App_65343_COMPARE-DATA-TOPIC'  # BuiltIn().get_variable_value("${gcp_pubsub_topic_id}")

    if 'Windows' == platform.system():
        publisher = pubsub_v1.PublisherClient.from_service_account_json(sa)
    else:
        publisher = pubsub_v1.PublisherClient()
    # The `topic_path` method creates a fully qualified identifier
    # in the form `projects/{project_id}/topics/{topic_id}`
    topic_path = publisher.topic_path(project_id, topic_id)
    # When you publish a message, the client returns a future.
    future = publisher.publish(topic_path, data)
    print(future.result())


def get_test_result_from_pubsub():
    project_id = config.gcp_project_id  # 'sbx-65343-autotest7-b-39c89c25'  # BuiltIn().get_variable_value("${gcp_pubsub_project_id}")
    subscription_id = config.pubsub_subscription_id  # "App_65343_dv_auto_test_sub"
    if 'Windows' == platform.system():
        subscriber_client = pubsub_v1.SubscriberClient.from_service_account_json(sa)
    else:
        subscriber_client = pubsub_v1.SubscriberClient()
    # existing subscription
    subscription = subscriber_client.subscription_path(project_id, subscription_id)

    def callback(message):
        print(message)
        message.ack()

    future = subscriber_client.subscribe(subscription, callback)

    try:
        # future.result()
        return future.result(timeout=5)
    except KeyboardInterrupt:
        future.cancel()


def get_test_result_from_pubsub2(msg_id):
    project_id = config.gcp_project_id  # 'sbx-65343-autotest7-b-39c89c25'  # BuiltIn().get_variable_value("${gcp_pubsub_project_id}")
    subscription_id = config.pubsub_subscription_id  # "App_65343_dv_auto_test_sub"
    if ('Windows' == platform.system()):
        subscriber = pubsub_v1.SubscriberClient.from_service_account_json(sa)
    else:
        subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(project_id, subscription_id)
    NUM_MESSAGES = 30
    retry_times = 0

    # Wrap the subscriber in a 'with' block to automatically call close() to
    # close the underlying gRPC channel when done.
    with subscriber:
        # The subscriber pulls a specific number of messages. The actual
        # number of messages pulled may be smaller than max_messages.
        response = None

        while (response is None or len(response.received_messages) == 0) and retry_times <= 10:
            time.sleep(3)
            response = subscriber.pull(
                request={"subscription": subscription_path, "max_messages": NUM_MESSAGES},
                retry=retry.Retry(deadline=300),
            )

            for received_message in response.received_messages:
                # print(f"Received: {received_message.message.data}.")
                msg_content = json.loads(received_message.message.data.decode('utf-8'))
                if msg_id == msg_content['id']:
                    ack_ids = [received_message.ack_id]
                    subscriber.acknowledge(request={"subscription": subscription_path, "ack_ids": ack_ids})
                    print(f"Received and acknowledged message: {msg_content} - from {subscription_path}.")
                    if not msg_content['status']:
                        exc_message = 'TestCase\t:' + BuiltIn().get_variable_value("${filters}", default='') + '\nErrorMessage:' + msg_content.get('fail_reason',
                                                                                                                                                     'The data of Dashboard is different with Data lake, Compare Failed') + "\n"
                        # add with index information for front user
                        if 'diff_index_json' in msg_content:
                            exc_message += "\n JSON diff index: " + '\t'.join([str(t) for t in msg_content.get('diff_index_json')])

                        if 'diff_index_bq' in msg_content:
                            exc_message += "\n BQ diff index: " + '\t'.join([str(t) for t in msg_content.get('diff_index_bq')])

                        raise Exception(exc_message)
                    else:
                        return

            retry_times = retry_times + 1
            response = None
    raise Exception("Cannot getting the data comparison result from GCP pubsub.")


# TODO: Refactor below function to compatible for both DV and DE
def get_test_result_from_pubsub_de(msg_id):
    project_id = config.gcp_project_id  # 'sbx-65343-autotest7-b-39c89c25'  # BuiltIn().get_variable_value("${gcp_pubsub_project_id}")
    subscription_id = config.pubsub_subscription_id_de  # "App_65343_dv_auto_test_sub"
    if 'Windows' == platform.system():
        subscriber = pubsub_v1.SubscriberClient.from_service_account_json(sa)
    else:
        subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(project_id, subscription_id)
    NUM_MESSAGES = 30
    retry_times = 0

    # Wrap the subscriber in a 'with' block to automatically call close() to
    # close the underlying gRPC channel when done.
    with subscriber:
        # The subscriber pulls a specific number of messages. The actual
        # number of messages pulled may be smaller than max_messages.
        response = None

        while (response is None or len(response.received_messages) == 0) and retry_times <= 10:
            time.sleep(3)
            response = subscriber.pull(
                request={"subscription": subscription_path, "max_messages": NUM_MESSAGES},
                retry=retry.Retry(deadline=300),
            )

            for received_message in response.received_messages:
                # print(f"Received: {received_message.message.data}.")
                msg_content = json.loads(received_message.message.data.decode('utf-8'))
                if msg_id == msg_content['id']:
                    ack_ids = [received_message.ack_id]
                    subscriber.acknowledge(request={"subscription": subscription_path, "ack_ids": ack_ids})
                    print(f"Received and acknowledged message: {msg_content} - from {subscription_path}.")
                    if not msg_content['status']:
                        raise Exception('TestCase\t:' + BuiltIn().get_variable_value("${filters}",
                                                                                     default='') + '\nErrorMessage:' + msg_content.get(
                            'fail_reason',
                            'The data of Dashboard is different with Data lake, Compare Failed'))
                    else:
                        return

            retry_times = retry_times + 1
            response = None
    raise Exception("Cannot getting the data comparison result from GCP pubsub.")


if __name__ == '__main__':
    # a()
    # b()
    # move_folder_to_folder_for_s32()
    # e()
    # f()
    # get_test_result_from_pubsub()
    # get_test_result_from_pubsub2()
    processing(r'C:\projects\code\Alice\dv-auto-test\core_framework\output\tmpdownloads',
               'd578e5c2-ba02-450e-94df-814de7c4a0e3.xlsx', True,
               '{"submissiondate123":"string", "count":"int"}',
               'select HireDate,CAST(New_Joiner AS STRING) AS New_Joiner,CAST(BadHires AS STRING) AS BadHires from `sbx-65343-autotest7-b-39c89c25`.autotest.hire')
    pass
