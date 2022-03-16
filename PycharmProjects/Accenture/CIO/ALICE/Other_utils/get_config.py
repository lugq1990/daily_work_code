# -*- coding:utf-8 -*-
"""
Created at 2:28 PM 10/24/2019
This is the common utils that could be used to get the config
object for project use case.
@author: guangqiang.lu
"""
import configparser
import os
import subprocess
import io


class Config(object):

    def _check_parameter(self):
        # if check the parameters
        if self.config_path is None:
            raise ValueError("You have to provide with config path!")

        if self.job_param_dict is not None:
            if not isinstance(self.job_param_dict, dict):
                raise TypeError("You have to provide with dictionary type parameters!")

        if self.config_in_hdfs:
            if self.sc is None:
                raise ValueError("When you want to use config in HDFS, you have to provide with spark context object!")

    def __init__(self, config_in_hdfs=False, config_path=None, job_name=None,  sc=None, job_param_dict=None):
        """
            Common function to get the config object. For reading HDFS file, I think with web HDFS should be fine.
            :param config_path: default is None.
                You have to provide this path! Even with HDFS or local path.
            :param job_name: default is None.
                What sub-job that could be used in later step to get the sub-job dictionaries.
                You could choose to provide or not.
            :param config_in_hdfs: Whether or not the config is in HDFS or not.
                If not, then we will try to
            :param sc: Spark Context object that we could interact with HDFS.
            :param job_param_dict: these parameters could be passed during the job run time.
            :return: config dictionary object that contain whole things in the config file.
                The logic is that first the get the job_param_dict as main config job, then
                with the sub_job config, the last is the app.conf. If there is same key, than
                just update the value with order.
                Order is: param_dict -> sub_job_dict -> app_dict
            """
        self.config_in_hdfs = config_in_hdfs
        self.config_path = config_path
        self.job_name = job_name
        self.sc = sc
        self.job_param_dict = job_param_dict
        self._check_parameter()
        self.app_config_name = 'app.conf'
        self.config = configparser.ConfigParser()
        self.config_dict = {}

    @staticmethod
    def _get_hdfs_files(hdfs_path):
        """
        This is just try to get the config files list.
        TODO: is there any a better way for this?
        :param hdfs_path: HDFS path
        :return: file list that file name endswith `conf`
        """
        file_list_cmd = "hdfs dfs -ls %s" % hdfs_path
        files_list = subprocess.check_output(file_list_cmd, shell=True).strip().decode('utf-8').split('\n')
        files_list = [x for x in files_list if 'conf' in x]
        files_list = [x.split(' ')[-1] for x in files_list]
        files_list = list(filter(lambda x: x.endswith('conf'), files_list))
        return files_list

    def _read_config_in_hdfs(self, config_hdfs_path):
        """
        This is just to read the config context in HDFS
        :param config_hdfs_path: HDFS config path that file exist
        :return: dictionary that contains in the config file.
        """
        try:
            string_conf = self.sc.textFile(config_hdfs_path).collect()
            buffer = '\n'.join(string_conf)
            self.config.read_string(buffer)
            return dict(self.config._sections)
        except Exception as e:
            raise IOError("When trying to get config object in HDFS, "
                          "we don't find the file %s in HDFS" % file)

    def _read_config_in_local(self, config_path):
        """
        This is to read config file in local server
        :param config_path: local server config file
        :return: dictionary that contains in the config file.
        """
        if not os.path.exists(config_path):
            raise IOError("We couldn't get the file: %s in local server." % config_path)

        try:
            self.config.read(config_path)
            return dict(self.config._sections)
        except Exception as e:
            raise io.FileIO("When we read in config in local path with error %s" % e)

    def get_config(self):
        """
        This is the main logic about the reading config both in HDFS and local server.
        :return: dictionary that contains the whole key-value pairs.
        """
        # We could get the dictionary file in HDFS or in local path
        if self.config_in_hdfs:
            # first should read the `app.conf` from HDFS
            if os.path.join(self.config_path, self.app_config_name) not in self._get_hdfs_files(self.config_path):
                raise io.FileIO("The `app.config` not in the HDFS path: %s, Please check first!" % self.config_path)

            app_config = self._read_config_in_hdfs(os.path.join(self.config_path, self.app_config_name))

            if app_config is not None and isinstance(app_config, dict):
                self.config_dict.update(app_config)

            if self.job_name is not None:
                # means that there is sub-job folder config need to be read. we should get it first.
                try:
                    # first to get file list in HDFS
                    file_lists = self._get_hdfs_files(os.path.join(self.config_path, self.job_name))
                    if len(file_lists) != 0:
                        # if and only if there exists config file object, then we could read the object
                        # then loop for each config object
                        for file in file_lists:
                            file_config = self._read_config_in_hdfs(file)

                            if (app_config is not None) and (isinstance(file_config, dict)):
                                self.config_dict.update(file_config)
                except Exception as e:
                    raise IOError("When we read the HDFS config with error: %s " % e)
            # the last time should update the job run time parameter
            if self.job_param_dict is not None:
                # as if I just update this whole job_param_dict, the whole things will be removed
                # so here add the logic to update with each keys in `job_param_dict`
                for key in self.job_param_dict.keys():
                    self.config_dict[key].update(self.job_param_dict[key])
        else:
            # the config file in local server
            app_config = self._read_config_in_local(os.path.join(self.config_path, self.app_config_name))

            if (app_config is not None) and (isinstance(app_config, dict)):
                self.config_dict.update(app_config)

            if self.job_name is not None:
                try:
                    sub_folder_path = os.path.join(self.config_path, self.job_name)
                    if not os.path.exists(sub_folder_path):
                        raise io.FileIO("We couldn't get the folder :%s" % sub_folder_path)

                    # get the whole things in sub-job folder
                    files_list = [x for x in os.listdir(sub_folder_path) if x.endswith('conf')]

                    if len(files_list) != 0:
                        for file in files_list:
                            file_config = self._read_config_in_local(os.path.join(sub_folder_path, file))

                            if file_config is not None and (isinstance(file_config, dict)):
                                self.config_dict.update(file_config)
                except Exception as e:
                    raise IOError("When we read the config file in local server with error: %s" % e)

            if self.job_param_dict is not None:
                # as if I just update this whole job_param_dict, the whole things will be removed
                # so here add the logic to update with each keys in `job_param_dict`
                for key in self.job_param_dict.keys():
                    self.config_dict[key].update(self.job_param_dict[key])

        return self.config_dict['config']


def _create_sc():
    import sys

    config = dict()
    config["spark_home"] = "/usr/hdp/current/spark2-client"
    config["pylib"] = "/python/lib"
    config['zip_list'] = ["/py4j-0.10.7-src.zip", "/pyspark.zip"]
    config['pyspark_python'] = "/anaconda-efs/sharedfiles/projects/mysched_9376/envs/cap_prd_py36_mml/bin/python"

    os.system("kinit -k -t /etc/security/keytabs/sa.mmld.mysched.keytab sa.mmld.mysched")
    os.environ["SPARK_HOME"] = config['spark_home']
    os.environ["PYSPARK_PYTHON"] = config["pyspark_python"]
    os.environ["PYLIB"] = os.environ["SPARK_HOME"] + config['pylib']
    zip_list = config['zip_list']
    for zip in zip_list:
        sys.path.insert(0, os.environ["PYLIB"] + zip)

    # This module must be imported after environment init.
    from pyspark.sql import SparkSession
    from pyspark import SparkConf

    conf = SparkConf().setAppName("create_mapping").setMaster("yarn")
    spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()

    return spark.sparkContext


def _read_config_example(use_hdfs=False, hdfs_path=None, sc=None):
    """
    This is just to show how to use the config object.
    I would recommend that no matter what parameters that
    you want to use, please convert the result type as you want
    in case with mistype problem.
    :param use_hdfs: Whether or not to use HDFS config
    :param hdfs_path: HDFS path of the config file
    :param sc: spark context that we could use to interact with HDFS.
    :return: None to return
    """
    param_config = {"config": {"env": "production"}}
    if use_hdfs:
        sc = _create_sc()
        print("read config from HDFS.")

        configer = Config(config_in_hdfs=True, config_path=hdfs_path, sc=sc)
        print("Without any change: get config:", configer.get_config())

        print('*'* 20)
        configer = Config(config_in_hdfs=True, config_path=hdfs_path, job_name='A001', sc=sc)
        print("Without sub-job, get config:", configer.get_config())

        print('*'*20)
        configer = Config(config_in_hdfs=True, config_path=hdfs_path, job_name='A001', job_param_dict=param_config, sc=sc)
        print("Without sub-job and parameter, get config:", configer.get_config())

    else:
        print("read config from local server")

        configer = Config(config_in_hdfs=False, config_path=".")
        print("Without any change: get config:", configer.get_config())

        print('*' * 20)
        configer = Config(config_in_hdfs=False, config_path=".", job_name='A001')
        print("Without sub-job, get config:", configer.get_config())

        print('*' * 20)
        configer = Config(config_in_hdfs=False, config_path=".", job_name='A001', job_param_dict=param_config)
        print("Without sub-job and parameter, get config:", configer.get_config())


if __name__ == '__main__':
    hdfs_path = '/data/discovery/mysched/config_test'
    _read_config_example(use_hdfs=False)
    _read_config_example(use_hdfs=True, hdfs_path=hdfs_path)
    
    config = _read_config_example(use_hdfs=False)
    env = config['env_info']['env']
    
    import pandas as pd
    
    
    def pas():
        df = pd.DataFrame()
        
        df.apply(lambda x: x == env)
