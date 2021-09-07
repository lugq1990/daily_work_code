# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
import os
import json
import numpy as np
import pandas as pd
import warnings

from google.cloud import bigquery

warnings.simplefilter('ignore')


# %%


# %% [markdown]
# Here should add a logic that we could do some other process logic like: <, >, ==, != etc.

# %%
from collections import Counter


def compare_main(df1, df2, cols=None, compare_op='=='):
    """Main compare logic for given columns.
    
    Support with <, >, !=, == etc. could be used for like others.
    """
    if not cols:
        # if `cols` is not provided, then we would try to get full columns
        cols = df1.columns
    
    if compare_op == '==':
        return compare_dfs_shape(df1, df2, cols=cols)
    elif compare_op != '==':
        # we could try to compare other operators logic here
        for col in cols:
            # DF should be checked first to satisfied dtype 
            tmp_df1 = df1[col]
            tmp_df2 = df2[col]
            # we could only support with `float64`, `int64` could be used for `<` etc.
            supported_dtypes = [np.float64, np.int64]
            tmp_df1_dtype = tmp_df1.dtypes.type
            tmp_df2_dtype = tmp_df2.dtypes.type
            
            if tmp_df1_dtype not in supported_dtypes or tmp_df2_dtype not in supported_dtypes:
                print("For {} operator, only float and int type is supported!".format(compare_op))
            
            if compare_op == "<":
                return (tmp_df1 < tmp_df2).all()
            elif compare_op == '>':
                return (tmp_df1 > tmp_df2).all()
            elif compare_op == '!=':
                return (tmp_df1 != tmp_df2).all()
    else:
        raise ValueError("Not supported operator: {}".format(compare_op))

    

def compare_dfs_shape(df1, df2, cols=None):
    """Compare two DFs is same or not?
    
    We need to handle NAN and duplicate records!
    """
    if not cols:
        cols = df1.columns
    
    # should based on each column
    is_same = True
    for col in cols:
        tmp_1 = df1[col].dropna()
        tmp_2 = df2[col].dropna()
        if len(tmp_1) != len(tmp_2):
            is_same = False
        counter1 = Counter(tmp_1).items()
        counter2 = Counter(tmp_2).items()
        if counter1 != counter2:
            is_same = False
        
        if not is_same:
            return False
        
    return is_same

# %% [markdown]
# ### Bigquery side
# 
# I think at least for now, **Bigquery** should be the base, and we could try to process data from DV part and try to process them like BQ type.

# %%
from pathlib import Path
cur_path = Path(__file__).parent

key_file = [x for x in os.listdir(cur_path) if x.endswith('json') and x.lower().startswith('cloud')][0]
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(cur_path, key_file)


# In case that we read files from current relative path, so let's change work folder.
os.chdir(cur_path)


# %%
client = bigquery.Client()


# %%
dataset_name = "auto_test"
table_name = "sample_data"

# dataset = client.get_dataset(dataset_name)
# table = dataset.table(table_name)

# make a easy SQL to get data from Bigquery
sample_sql = "select * from {}.{}".format(dataset_name, table_name)

query = client.query(sample_sql)
result = query.result()

table_schema_list = [(s.name, s.field_type) for s in result.schema]

for n, t in table_schema_list:
    print("Name: {}, type: {}".format(n, t))

df_bq = query.to_dataframe()

df_bq.head()

# %% [markdown]
# ### Load local JSON file like stream JSON string
# 
# 

# %%
df = pd.read_excel('sample.xlsx', engine='openpyxl')


# %%
# make column `c f g` to string for converting
df['c'] = df['c'].map(lambda x: str(x*100) + "%")
df['f'] = df['f'].map(lambda x: str(x)[:2] + ',' + str(x)[2:])
df['g'] = df['g'].map(lambda x: '/'.join(str(x.strftime('%Y-%m-%d')).split('-')))

df.head()


# %%
# write back into server and read it from local as a stream
json_df = df.to_json()

json_file_name = "sample.json"
with open(json_file_name, 'w') as f:
    json.dump(json_df, f)


# %%
# read from json to make pandas to infer it
with open(json_file_name, 'r') as f:
    json_data = json.load(f)


# %%
df_json = pd.read_json(json_data)

df_json.columns = df_bq.columns

df_json.head()


# %%
df_json.dtypes


# %%
df_bq.dtypes

# %% [markdown]
# We could find that if origin data is basic data structure, then for **NULL** will be ignored and pandas will infer correctly, if original data is string type, then **NULL** will be converted into **NONE** based on pandas, this should be taken care.
# 
# #### Noted
# Don't need to get out of **Object** type as if data in BQ is object, then it's string, then we should do compare content of each DataFrame!

# %%
# get two dataframe's data types, and get the same data type's columns
json_dtype = dict(df_json.dtypes)
bq_dtype = dict(df_bq.dtypes)

same_dtype_cols = []
other_dtype_cols = []
for k, _ in json_dtype.items():
    if json_dtype[k] == bq_dtype[k]:
        same_dtype_cols.append(k)
    else:
        other_dtype_cols.append(k)
        
print("Same data type columns: {}".format('\t'.join(same_dtype_cols)))
print("Diff data type columns: {}".format('\t'.join(other_dtype_cols)))


# %%
# First try to compore same data type column for these 2 DFs.
def compare_same_types(same_dtype_cols):
    same_df_json = df_json[same_dtype_cols]
    same_df_bq = df_bq[same_dtype_cols]

    return compare_dfs_shape(same_df_json, same_df_bq)


# %%
compare_same_types(same_dtype_cols)

# %% [markdown]
# ##### Process with not same type's columns

# %%
# For the other not same column then we need to try to process them each column directly
df_json[other_dtype_cols]


# %%
df_bq[other_dtype_cols]

# %% [markdown]
# #### 1. Datetime type from BQ

# %%
def compare_date_columns(df_bq=df_bq, df_json=df_json):
    # If both of them are date type, then we could use pandas.to_datetime try to convert them into a normal datetime, , and they will be same,
    # otherwise we will get error then should be False returned.
    
    date_cols = [k for k, v in bq_dtype.items() if v.name.startswith('datetime64')]
    date_df_bq = df_bq[date_cols]
    date_df_json = df_json[date_cols]
    
    if date_df_bq.shape != date_df_json.shape:
        return False
    
    sati = True
    try:
        # it's fine if we have many columns by using `apply`
        date_df_bq = date_df_bq.apply(pd.to_datetime)
    except:
        sati = False
    
    try:
        date_df_json = date_df_json.apply(pd.to_datetime)
    except:
        sati = False
    
    if not sati:
        return False
    
    
    return compare_dfs_shape(date_df_bq, date_df_json)

compare_date_columns()

# %% [markdown]
# ##### 2. Compare other types
# 
# We could just try to compare others with string types will be fine, just remove some special characters, and compare them.
# We need to process each column with some pre-defined rules to compare! Like: `%`, `,`, `&`, `$`, etc.
# 
# I think except for special `%`, others could just with replacement will be fine.

# %%
date_cols = [k for k, v in bq_dtype.items() if v.name.startswith('datetime64')]
other_not_sati_cols = set(list(bq_dtype.keys())) - set(date_cols) - set(same_dtype_cols)
other_not_sati_cols


# %%
other_sati_json_df = df_json[other_not_sati_cols]
other_sati_bq_df = df_bq[other_not_sati_cols]

print(other_sati_json_df.head())
print(other_sati_bq_df.head())


# %%
# This should base on JSON data only, as BQ won't accept this.
# We could try to convert full columns into `string`, and try to get
# special: % from dataframe

def compare_percen_data(other_sati_json_df=other_sati_json_df, 
                        other_sati_bq_df=other_sati_bq_df,
                        float_round_estimation = 4,
                        per_threshould = .9):
    other_sati_json_df = other_sati_json_df.astype(str)

    # Loop each columns to get percentage columns.
    percen_cols = []
    for col in other_sati_json_df.columns:
        per_num = other_sati_json_df[col].map(lambda x: True if "%" in x else False).sum()
        null_num = other_sati_json_df[col].isnull().sum()
        if per_num:
            if null_num:
                if per_num / (null_num + per_num) >= per_threshould:
                    percen_cols.append(col)
            else:
                if per_num / len(other_sati_json_df) >= per_threshould:
                    percen_cols.append(col)

    # If we have get percentage columns, then need to convert them into float
    other_sati_json_df[percen_cols] = other_sati_json_df[percen_cols].applymap(lambda x: float(x.replace('%', ''))/100)
    
    # Key notes here: WE SHOULDN'T COMPARE FLOAT, SHOULD CONVERT INTO STRING!
    # convert BQ df either, so could compare easy...Let's just hard-code this for 4-digits to keep
    per_convert_json = other_sati_json_df[percen_cols].applymap(lambda x: "%.4f" %  round(x, float_round_estimation))
    per_convert_bq = other_sati_bq_df[percen_cols].applymap(lambda x: "%.4f" % round(x, float_round_estimation))


    return compare_dfs_shape(per_convert_json, per_convert_bq), percen_cols

per_compare_res, per_cols = compare_percen_data()
print(per_compare_res)


# %%
# LET'S PROCESS OTHER TYPES DATA.
# remove full special characters that we may face.
import re
import string
special_characters = re.escape(string.punctuation)

def compare_other_sati_columns(other_sati_bq_df=other_sati_bq_df,
                              other_sati_json_df=other_sati_json_df):
    other_str_cols = set(list(other_sati_bq_df.columns)) - set(per_cols)

    def remove_spe_cha(x):
        return re.sub(r"[" + special_characters + "]", "", str(x))

    other_str_bq_df = other_sati_bq_df[other_str_cols].applymap(lambda x: remove_spe_cha(x))
    other_str_json_df = other_sati_json_df[other_str_cols].applymap(lambda x: remove_spe_cha(x))

    return compare_dfs_shape(other_str_bq_df, other_str_json_df)
    
    
compare_other_sati_columns()


# %%



