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


class DataProcessingLibrary:
    ROBOT_LIBRARY_SCOPE = 'GLOBAL'
    __version__ = '0.1'
    downloads_path = None

    @keyword
    def test(self, downloads_path=None):
        pass
