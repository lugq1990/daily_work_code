#!/usr/bin/env python
import sys
from robot import run_cli, rebot_cli
import datetime
import os
import shutil
import devops_api_utils
import dry_run_util
import config

os.environ['GOOGLE_CLOUD_PROJECT'] = config.gcp_project_id

cur_dt = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
test_result_repo = os.path.join(os.getcwd(), 'test_result_repo')
output_dir = os.path.join(os.getcwd(), 'output')
xunit_file = os.path.join('xunit_files', 'xunit_%s.xml' % cur_dt)
dryrun_xunit_file = os.path.join('xunit_files', 'xunit_dryrun.xml')


def main():
    common = ['--pythonpath', 'ud_keywords', '--log', 'log.html', '--loglevel', 'TRACE', '--report', 'report.xml', '--outputdir', output_dir]
    xunit = ['--xunit', dryrun_xunit_file if '--dryrun' in sys.argv[1:] else xunit_file]

    common = common + xunit
    conf = ['--variable', 'BROWSER:headlesschrome', '--variable', 'TMPDOWNLOADS:%s' % os.path.join(output_dir, 'tmpdownloads'), '--variable',
            'XUNIT:%s' % os.path.join(output_dir, xunit_file)]
    args = conf + common + sys.argv[1:]
    print(args)
    run_cli(args, exit=False)

    # push_test_result_to_git_repo()
    if '--dryrun' in common:
        dry_run_util.check_dry_run_result(os.path.join(output_dir, xunit_file))
    else:
        # devops_api_utils.create_test_run_by_xunit(os.path.join(output_dir, xunit_file), os.path.join(output_dir, 'log.html'))
        pass


# run_cli(['--name', 'IE', '--variable', 'BROWSER:IE', '--output', 'out/ie.xml'] + common, exit=False)
# rebot_cli(['--name', 'Login', '--outputdir', 'out', 'out/fx.xml', 'out/ie.xml'])

def push_test_result_to_git_repo():
    from git import Repo
    test_repo = test_result_repo
    if os.path.exists(test_repo):
        del_file(test_repo)

    git_ssh_identity_file = r'C:/projects/code/Alice/robot_auto_test/id_rsa'
    git_ssh_cmd = 'ssh -o StrictHostKeyChecking=no -i "%s"' % git_ssh_identity_file
    # git@ssh.dev.azure.com:v3/accenturecio02/AIA0038Incubator_65343/dv-auto-test
    repo = Repo.clone_from('git@ssh.dev.azure.com:v3/RRHB/DV-AUTO-TEST/DV-AUTO-TEST', test_repo, env=dict(GIT_SSH_COMMAND=git_ssh_cmd), branch='main')

    shutil.copy(os.path.join(output_dir, xunit_file), test_repo)
    remote = repo.remote()
    remote.pull()
    index = repo.index
    index.add(['*.xml'])
    index.commit('this is a test')
    remote.push()


def del_file(root_path):
    for i in os.listdir(root_path):
        file_data = os.path.join(root_path, i)
        if os.path.isfile(file_data):
            os.chmod(file_data, 0o777)
            os.remove(file_data)
        elif os:
            del_file(file_data)
    shutil.rmtree(root_path)


if __name__ == '__main__':
    main()
