from azure.devops.connection import Connection
from msrest.authentication import BasicAuthentication
from azure.devops.v6_0.test_plan import TestPlanClient
from azure.devops.v6_0.test import TestClient
from azure.devops.v6_0.test.models import RunCreateModel, TestCaseResult, RunCreateModel, TestAttachmentRequestModel
from azure.devops.v6_0.work_item_tracking import WorkItemTrackingClient
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from io import StringIO
import pickle
import datetime
import base64
import json
import config


class RobotTestCase():
    def __init__(self, id, name, steps, params):
        self.id = id
        self.name = name
        self.steps = steps
        self.params = params


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, d):
        self.text.write(d)

    def get_data(self):
        return self.text.getvalue()


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


# Fill in with your personal access token and org URL

personal_access_token = '#{APP_PAT}#'

# organization_url = 'https://dev.azure.com/accenturecio02'
# project_id = "AIA0038Incubator_65343"

# organization_url = 'https://dev.azure.com/{}'.format('#{ADO_ORGANIZATION}#')
# project_id = '#{ADO_PROJECT_ID}#'

TEST_RUN_ID = 0
TEST_POINT_RESULT_DICT = {}
TEST_POINT_CASE_DICT = {}
SUB_TEST_SUIT_LST = None

organization_url = config.organization_url
project_id = config.project_id

plan_id = config.plan_id
suite_id = config.suite_id

# plan_id = 178260
# suite_id = 178261
# plan_id = 208990
# suite_id = 208991

run_name = "TestRun_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')

# Create a connection to the org
credentials = BasicAuthentication('', personal_access_token)
connection = Connection(base_url=organization_url, creds=credentials)


def refresh_test_variables():
    global organization_url, project_id, plan_id, suite_id, connection
    organization_url = config.organization_url
    project_id = config.project_id

    plan_id = config.plan_id
    suite_id = config.suite_id
    connection = Connection(base_url=organization_url, creds=credentials)


def get_test_cases(with_steps=True):
    global SUB_TEST_SUIT_LST
    ret = []
    test_plan_client = connection.clients_v6_0.get_test_plan_client()  # type: TestPlanClient

    # test_suites = test_plan_client.get_test_suites_for_plan(project_id, plan_id, expand='Children')

    def _make_test_cases(suite, test_case_list):
        test_cases = []
        for test_case in test_case_list:
            if test_case.work_item.name.startswith('.'):
                continue
            if test_case.work_item.work_item_fields:
                steps_xml = get_xml_from_test_case(test_case, 'Microsoft.VSTS.TCM.Steps')

                if steps_xml:
                    steps = get_steps(steps_xml)
                else:
                    steps = None

                parameter_xml = get_xml_from_test_case(test_case, 'Microsoft.VSTS.TCM.LocalDataSource')
                if parameter_xml:
                    param_names, param_datas = get_shared_parameters(parameter_xml)
                    print(param_names, param_datas)
                else:
                    param_names = None
                    param_datas = None

                if with_steps:
                    test_cases.append(RobotTestCase(test_case.work_item.id, test_case.work_item.name, steps, param_datas))
                else:
                    test_cases.append(RobotTestCase(test_case.work_item.id, test_case.work_item.name, None, None))
        ret.append({"test_suite": suite, "test_cases": test_cases})

    def _get_testcases_from_test_suites(proj_id, p_id, s_id, p_s_name=''):
        ts = test_plan_client.get_test_suite_by_id(proj_id, p_id, s_id, expand='Children')
        test_case_list = test_plan_client.get_test_case_list(project_id, plan_id, ts.id)
        # if s_name == '':
        #    s_name = ts.name + '-' + str(ts.id) + '`|'
        if len(test_case_list) > 0:
            _make_test_cases({"suite_id": ts.id, "suite_name": p_s_name + ts.name + '-' + str(ts.id)}, test_case_list)

        if ts.has_children:
            p_s_name = p_s_name + ts.name + '-' + str(ts.id) + '`|'
            for suite in ts.children:
                if suite.name.startswith("."):
                    continue
                _get_testcases_from_test_suites(proj_id, p_id, suite.id, p_s_name=p_s_name)

    _get_testcases_from_test_suites(project_id, plan_id, suite_id)
    '''         
    if len(test_suite.children) == 0:
        SUB_TEST_SUIT_LST = [suite_id]
    else:
        SUB_TEST_SUIT_LST = [{"suite_id": suite.id, "suite_name": suite.name} for suite in test_suite.children if not suite.name.startswith(".")]

    for test_suite in SUB_TEST_SUIT_LST:
        test_case_lst = test_plan_client.get_test_case_list(project_id, plan_id, test_suite['suite_id'])
        _make_test_cases(test_suite, test_case_lst)

    # 7/16/2021 get the test cases from root suite - start
    test_case_lst = test_plan_client.get_test_case_list(project_id, plan_id, suite_id)
    if len(test_case_lst) > 0:
        _make_test_cases({"suite_id": test_suite.id, "suite_name": test_suite.name}, test_case_lst)
    # 7/16/2021 get the test cases from root suite - end 
    '''
    with open("test_metadata.txt", 'wb') as f:  # 打开文件
        pickle.dump(ret, f)
    return ret


def create_test_run_by_xunit(file_path, log_path=None):
    global TEST_RUN_ID
    global TEST_POINT_RESULT_DICT
    global TEST_POINT_CASE_DICT
    if TEST_RUN_ID != 0:
        return
    tree = ET.parse(file_path)
    rf_test_results = tree.getroot()
    # todo add suite filter

    with open('test_metadata.txt', 'rb') as f:
        test_metadata = pickle.load(f)

    run_tp_metadata = []
    for metadata in test_metadata:
        test_suite_id, test_suite_name = metadata['test_suite'].values()

        ado_test_points = get_test_points_lst(test_suite_id)
        for test_case in rf_test_results:
            for test_point in ado_test_points:
                if test_case.attrib['name'] == "{}-{}-{}".format(test_point['test_case_name'], test_point['test_case_id'], test_suite_id):
                    TEST_POINT_CASE_DICT[test_case.attrib['name']] = str(test_point['id'])
                    # outcome = "FAILED" if len(test_case) > 0 and test_case[0].tag.upper() == "FAILURE" else "PASSED"
                    # error_message = test_case[0].attrib['message'] if outcome == "FAILED" else ""
                    item = {"id": str(test_point['id']),
                            # "duration_in_ms": 15000,
                            "outcome": 'NotExecuted',
                            # "error_message": '',
                            }
                    run_tp_metadata.append(item)
                    break

    # initialize metadata of test run
    metadata_run = {
        "name": run_name,
        "plan": {"id": plan_id},
        # "state": "NotStarted",
        "point_ids": [tp['id'] for tp in run_tp_metadata]
    }

    ado_update_test_result_lst = []
    ado_test_run = create_test_run(RunCreateModel(**metadata_run))
    # return
    TEST_RUN_ID = ado_test_run.id
    ado_test_results = get_test_results(ado_test_run.id)

    for test_result in ado_test_results:
        for tp_item in run_tp_metadata:
            if tp_item['id'] == test_result.test_point.id:
                tp_item['id'] = test_result.id
                TEST_POINT_RESULT_DICT[str(test_result.test_point.id)] = test_result.id
                ado_update_test_result_lst.append(tp_item)
                break

    # create_run_attachment(log_path, ado_test_run.id)
    # print(ado_update_test_result_lst)
    update_test_results(ado_update_test_result_lst)


def finalize_test_run(log_path, err_msg=''):
    create_run_attachment(log_path, TEST_RUN_ID)
    metadata_run = {
        "state": "Completed",
        "completedDate": datetime.datetime.now().strftime('%Y-%m-%d'),
        "errorMessage": err_msg
    }
    update_test_run(metadata_run)


def get_test_points_lst(suite_id):
    test_plan_client = connection.clients_v6_0.get_test_plan_client()  # type: TestPlanClient
    test_points = test_plan_client.get_points_list(project_id, plan_id, suite_id)
    return [{"id": test_point.id, "test_case_name": test_point.test_case_reference.name, "test_case_id": test_point.test_case_reference.id} for test_point in test_points]


def create_test_run(model_test_run):
    test_client = connection.clients_v6_0.get_test_client()  # type: TestClient
    test_run = test_client.create_test_run(model_test_run, project_id)
    return test_run


def update_test_run(model_test_run):
    test_client = connection.clients_v6_0.get_test_client()  # type: TestClient
    test_run = test_client.update_test_run(model_test_run, project_id, TEST_RUN_ID)
    return test_run


def get_test_results(run_id):
    test_client = connection.clients_v6_0.get_test_client()  # type: TestClient
    test_results = test_client.get_test_results(project_id, run_id)
    return test_results


def update_test_results(model_results):
    test_client = connection.clients_v6_0.get_test_client()  # type: TestClient
    print(str(TEST_RUN_ID))
    test_client.update_test_results(model_results, project_id, TEST_RUN_ID)


def get_test_result_id_by_test_case(test_case_name):
    test_point_id = TEST_POINT_CASE_DICT.get(test_case_name)
    return None if test_point_id is None else TEST_POINT_RESULT_DICT.get(test_point_id)


# create_test_run_by_xunit('xunit_20210511070358011542.xml')


def create_run_attachment(file_path, run_id):
    with open(file_path, 'rb') as f:
        binary_file_data = f.read()
        base64_encoded_data = base64.b64encode(binary_file_data)
        base64_message = base64_encoded_data.decode('utf-8')

    model_run_attachment = {"stream": base64_message,
                            "file_name": "log.html",
                            "comment": "Test attachment upload",
                            "attachment_type": "GeneralAttachment"
                            }

    test_client = connection.clients_v6_0.get_test_client()  # type: TestClient
    test_run_attachment = test_client.create_test_run_attachment(TestAttachmentRequestModel(**model_run_attachment), project_id, run_id)
    if not test_run_attachment:
        return None

    return test_run_attachment.id


def get_xml_from_test_case(test_case, key):
    try:
        index = [val for item in test_case.work_item.work_item_fields for val in item.keys()].index(key)
    except ValueError:
        index = -1

    if index != -1:
        return test_case.work_item.work_item_fields[index].get(key)
    else:
        return None


def get_steps(steps_xml):
    steps_root = ET.ElementTree(ET.fromstring(steps_xml)).getroot()
    return [strip_tags(step[0].text) for step in steps_root]


def get_shared_parameters(parameter_xml):
    datas = []

    shared_parameter = json.loads(parameter_xml)
    parameter_id = shared_parameter.get('sharedParameterDataSetIds')[0]

    work_item_client = connection.clients_v6_0.get_work_item_tracking_client()  # type: WorkItemTrackingClient
    wi = work_item_client.get_work_item(parameter_id, project_id)

    xml = wi.fields.get('Microsoft.VSTS.TCM.Parameters')
    parameter_root = ET.ElementTree(ET.fromstring(xml)).getroot()

    names = [param.text for param in parameter_root.find('paramNames').findall('param') if param.text]

    data_rows = [dataRow for dataRow in parameter_root.find('paramData')]
    for row in data_rows:
        data_item = {}
        kvps = row.findall('kvp')
        for kvp in kvps:
            data_item.update({kvp.attrib['key']: strip_tags(kvp.attrib['value'])})
            if len(kvps) == len(data_item.keys()):
                for key in list(data_item.keys())[:-1]:
                    condition = ','.join(["'{}'".format(val) for val in data_item[key].split(',')])
                    # data_item['Query'] = data_item['Query'].replace('@{}'.format(key), condition)
        datas.append(data_item)
    return names, datas
    #
    # for data in ret:
    #     print(data)


def get_related_work(workitem_id):
    work_item_client = connection.clients_v6_0.get_work_item_tracking_client()  # type: WorkItemTrackingClient
    wi = work_item_client.get_work_item(workitem_id, project_id, expand='Relations')


def get_variable_group():
    group_id = config.variable_group_id
    test_agent_client = connection.clients_v6_0.get_task_agent_client()  # type: TaskAgentClient
    vg = test_agent_client.get_variable_group(project_id, group_id)
    if vg:
        # a = {key:_format_value(vg.variables[key].value) for key in vg.variables.keys()}
        return {key: _format_value(vg.variables[key].value) for key in vg.variables.keys()}
    else:
        return None


def _format_value(value):
    try:
        json_object = json.loads(value)
        return json_object
    except ValueError as e:
        return value


def init_configuration():
    vars_group = get_variable_group()
    config.plan_id = vars_group['ADO_PLAN_ID']
    config.suite_id = vars_group['ADO_SUITE_ID']
    refresh_test_variables()


init_configuration()

if __name__ == "__main__":
    # ret = get_test_cases()
    create_test_run_by_xunit('xunit_dryrun.xml')
    # time.sleep(10)
    # finalize_test_run("123", 'Error happens')

    print("aa")
    # get_shared_parameters(179100)
    # get_related_work(204021)
