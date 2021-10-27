import xml.etree.ElementTree as ET
import sys


def check_dry_run_result(file_path):
    # tree = ET.parse(r'C:\projects\code\Alice\dv-auto-test\output\xunit_files\xunit_20210513064357731346.xml')
    tree = ET.parse(file_path)
    rf_dry_run_results = tree.getroot()

    dry_run_error = []
    for test_case in rf_dry_run_results:
        if len(test_case) > 0 and test_case[0].tag.upper() == 'FAILURE':
            item = {"test_case": test_case.attrib['name'], "error_message": test_case[0].attrib['message']}
            dry_run_error.append(item)

    if len(dry_run_error) == 0:
        print("No Syntax Error Found in Test Cases......")
        sys.exit(0)
    else:
        print("Syntax errors found in Test Cases in below:")
        for item in dry_run_error:
            print('Test Case - {}'.format(item['test_case']))
            if '\n' in item['error_message']:
                for error in item['error_message'].split('\n'):
                    if len(error) > 0 and error[0].isdigit():
                        print(' ' * 4 + error)
            else:
                print(' ' * 4 + item['error_message'])
        sys.exit(-1)


if __name__ == "__main__":
    check_dry_run_result('')
