#!/usr/bin/env python

import devops_api_utils
import os
import config
import json
import shutil
from urllib.parse import quote

settings = ['*** Settings ***\n'
            'Documentation     description of the test\n',
            'Resource          ud_resource/resource.robot\n',
            'Resource          ud_resource/conf.robot\n',
            'Resource          ud_resource/seleow.robot\n',
            'Resource          ud_resource/dataprocessing.robot\n',
            'Library  SeleniumLibraryExtension    run_on_failure=Nothing\n',
            # 'Suite Setup    Open Browser And Login Dashboard\n',
            'Suite Teardown    Close Browser\n',
            # 'Test Teardown     Close Browser\n'
            ]

settings = settings + ['\n', '*** Test Cases ***\n']


def get_test_cases_from_devops():
    test_suites = devops_api_utils.get_test_cases()
    print(test_suites)
    if len(test_suites) > 0:
        shutil.rmtree('devops_testcases/', ignore_errors=True)
    for ts in test_suites:
        suite_id = ts['test_suite']['suite_id']
        suite_name = ts['test_suite']['suite_name']
        test_cases = ts['test_cases']
        test_case = []
        if len(test_cases) == 0:
            continue
        for tc in test_cases:
            # transform_2_robot(tc)
            if len(tc.steps) == 0:
                # raise Exception('No any steps in TestCase %s' % test_case.name)
                continue
            test_case.append('\n' + tc.name + '-' + str(tc.id) + '-' + str(suite_id) + '\n')
            obj_id = None

            steps = tc.steps

            for step in steps:
                if step.lower().find('Get Target Obj') != -1:
                    obj_id = step.split('::')[-1]

            params = tc.params
            shared_param_kv = {}
            if params is not None:
                for p in params:
                    k = p.get('Query_Index', None)
                    shared_param_kv[k] = p['Query_Content'] if k is not None else None

            template_steps = parse_params_1(params, obj_id)

            def check_comment(step_str, i=0):
                position = step_str.find('#', i)
                if position > 0:
                    if step_str[position - 1] == '\\':
                        i = position + 1
                        return check_comment(step_str, i)
                    else:
                        return position
                else:
                    return 0

            for step in steps:
                step = step.replace('\n', ' ').replace(' ', ' ')
                comment_start = check_comment(step)
                if comment_start > 0:
                    # step = step[0:comment_start] + '  ' + step[comment_start:]
                    step = step[0:comment_start]
                if step.lower().replace(' ', '').startswith('get expect and actual data and compare them'.replace(' ', '')):
                    test_case = test_case + template_steps
                elif step.lower().replace(' ', '').startswith('Expect Actual Data Equal Expected Data'.lower().replace(' ', '')):
                    step = "Run Keyword And Continue On Failure  " + step
                    test_case.append('    ' + step + '\n')
                else:
                    if step.find('::') != -1:
                        keyword = step[:step.find('::')].strip()
                        while keyword.find('  ') != -1:  # Replace two blank to one
                            keyword = keyword.replace('  ', ' ')
                        arguments = step[step.find('::') + 2:].strip()
                        arguments = ' '.join(arguments.split())
                        var_group = devops_api_utils.get_variable_group()
                        for key, value in var_group.items():
                            arguments = arguments.replace("@{%s}" % key, str(value))
                        if keyword.lower().find('get actual data byfilter') != -1:
                            if not arguments.startswith('select=$::'):
                                if not arguments.startswith('{'):
                                    arguments = '{' + arguments + '}'
                                arguments = arguments.replace('\\,', '%2C').replace('&', '%26')
                                argument_json = json.loads(arguments)
                                if argument_json is None:
                                    continue
                                filters = []
                                for k, v in argument_json.items():
                                    filters.append('select=$::{},{}'.format(k, v))
                                arguments = '&'.join(filters)

                        if arguments.startswith('${') and arguments.endswith('}'):
                            arguments = arguments[2:-1]
                            arguments = shared_param_kv[arguments]

                        step = keyword + '    ' + arguments
                    test_case.append('    ' + step + '\n')

        test_case_file_content = settings + test_case
        suite_folders = suite_name.replace('`|', os.path.sep)
        robot_folder = os.path.join('devops_testcases', suite_folders)
        robot_file = os.path.join(robot_folder, 'test_case.robot')
        if not os.path.exists(robot_folder):
            os.makedirs(robot_folder)

        with open(robot_file, 'w', encoding='utf-8') as f:
            f.writelines(test_case_file_content)


def parse_params_1(params, obj_id):
    template_steps = []
    if params is not None and obj_id is not None:
        template_steps.append('    @{param_list} =    Create List')
        for p in params:
            qlik_obj_id = p['QlikSense_ObjId']
            if obj_id != qlik_obj_id:
                continue
            filter_keys = [key for key in p.keys() if key.lower().startswith('filter')]
            filters = []
            for fk in filter_keys:
                fv = p[fk]
                fv = fv.replace(' ', '${SPACE}').replace('Filter_', '')
                filters.append('select=$::{},{}'.format(fk, fv))

            query_keys = [key for key in p.keys() if key.lower().startswith('query')]
            queries = []
            for qk in query_keys:
                qv = p[qk]
                qv = qv.replace(' ', '${SPACE}')
                queries.append(qv)

            template_steps.append("    " + "{}`$`{}`$`{}`$`{}".format(qlik_obj_id, "&".join(filters), ";".join(queries), 'schema'))
        template_steps.append("\n    FOR    ${p}    IN    @{param_list}\n")
        template_steps.append("        Run Keyword And Continue On Failure  Get Actual and Expect data and Compare them    ${p}\n")
        template_steps.append("    END\n")
    return template_steps


def transform_2_robot(test_case):
    if len(test_case.steps) == 0:
        # raise Exception('No any steps in TestCase %s' % test_case.name)
        pass
    settings.append(test_case.name + '\n')
    steps = test_case.steps
    for step in steps:
        settings.append('    ' + step + '\n')


if __name__ == '__main__':
    get_test_cases_from_devops()
    print(settings)
