#!/usr/bin/env python

from robot.api.deco import keyword, library
from robot.running import Keyword, RUN_KW_REGISTER


def run_keyword_variant(resolve):
    def decorator(method):
        RUN_KW_REGISTER.register_run_keyword('BuiltIn', method.__name__,
                                             resolve, deprecation_warning=False)
        return method

    return decorator


class StringLibraryExtension:
    ROBOT_LIBRARY_SCOPE = 'GLOBAL'
    __version__ = '0.1'
    downloads_path = None

    @keyword('Find ${str1} in ${str2}')
    def find(self, str1, str2):
        i = str2.find(str1)
        if i == -1:
            return False
        else:
            return True

    @keyword
    def get_single_obj_url(self, dashboard_url, obj_id, filters=''):
        start_str = 'app/'
        end_str = '/sheet'
        app_id = dashboard_url[dashboard_url.find(start_str) + len(start_str): dashboard_url.find(end_str)]
        url = "https://bi8.ciostage.accenture.com/azure/single/?appid={}&obj={}&opt=ctxmenu,currsel&{}".format(app_id, obj_id, filters)
        return url

    @keyword
    def join_string(self, list_str, sep_str=''):
        return sep_str.join(list_str)
