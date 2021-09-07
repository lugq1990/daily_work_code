# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import datetime
import os
import copy

path = 'C:/Users/guangqiiang.lu/Documents/lugq/workings/201903/hp'
org_file = 'org2.xlsx'
add_file = 'add2.xlsx'


org_df = pd.read_excel(os.path.join(path, org_file), converters={'date':pd.to_datetime})
add_df = pd.read_excel(os.path.join(path, add_file), converters={'End Time': pd.to_datetime})


test_df = copy.copy(org_df)
test_cols = [str(x) for x in test_df.columns]
test_df.columns = test_cols

test_df['60131720'].head()

