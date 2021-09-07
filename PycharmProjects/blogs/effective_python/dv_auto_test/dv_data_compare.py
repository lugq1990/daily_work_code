import os
import json
import numpy as np
import pandas as pd
import warnings
import re
import string
from collections import Counter, defaultdict
import base64
from decimal import Decimal, localcontext, ROUND_HALF_UP

from google.cloud import bigquery
from google.cloud import pubsub_v1

warnings.simplefilter('ignore')


def compare_main(df1, df2, cols=None, compare_op='=='):
    """Main compare logic for given columns.
    
    Support with <, >, !=, == etc. could be used for like others.
    """
    if cols is None:
        # if `cols` is not provided, then we would try to get full columns
        cols = df1.columns
    
    if isinstance(cols, str):
        # in case we just get a string column name.
        cols = [cols]
    
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
                print("For {} operator, only float and int type is supported! But get type:{}!".format(compare_op, tmp_df1_dtype))
            
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
    if cols is None:
        cols = df1.columns
        
    # should based on each column
    # This should base on each row into a tuple to compare.
    
    is_same = True
    
    # first let's try to make it into a array
    val1 = df1[cols].dropna().values
    val2 = df2[cols].dropna().values
    
    if val1.shape != val2.shape:
        is_same = False
    
    if not is_same:
        # if there are not same, just return, no need to do comparation for cols.
        return is_same
    
    # loop for each row.
    res1 = defaultdict(list)
    res2 = defaultdict(list)
    for i in range(len(val1)):
        row1 = tuple(val1[i])
        row2 = tuple(val2[i])
        if row1 not in res1:
            res1[row1] = 1
        else:
            res1[row1] += 1
            
        if row2 not in res2:
            res2[row2] = 1
        else:
            res2[row2] += 1
    
    # try to compare these two dictionaries.
    if res1.items() != res2.items():
        is_same = False
        
    return is_same

def get_bq_df(query):
    client = bigquery.Client()

    # dataset_name = "auto_test"
    # table_name = "sample_data"

    # # dataset = client.get_dataset(dataset_name)
    # # table = dataset.table(table_name)

    # # make a easy SQL to get data from Bigquery
    # sample_sql = "select * from {}.{}".format(dataset_name, table_name)

    query = client.query(query)
    result = query.result()

    table_schema_list = [(s.name, s.field_type) for s in result.schema]

    for n, t in table_schema_list:
        print("Name: {}, type: {}".format(n, t))

    df_bq = query.to_dataframe()

    return df_bq


def get_json_df(data):
    # TODO: DF from JSON should be added here
    # data = json.loads(base64.b64decode(event['data']).decode('utf-8'))
    return pd.DataFrame(data)


def get_same_diff_type_cols(df_json, df_bq):
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

    return same_dtype_cols, other_dtype_cols


def convert_date_cols(df_json, df_bq):
    # If both of them are date type, then we could use pandas.to_datetime try to convert them into a normal datetime, , and they will be same,
    # otherwise we will get error then should be False returned.
    
    date_cols = [k for k, v in dict(df_bq.dtypes).items() if v.name.startswith('datetime64')]
#     date_df_bq = df_bq[date_cols]
#     date_df_json = df_json[date_cols]
    
#     if date_df_bq.shape != date_df_json.shape:
#         return False
    
    if len(date_cols) == 0:
        return df_json, df_bq
    
    print("Get Datetype columns: {} to process.".format('\t'.join(date_cols)))
    sati = True
    try:
        # it's fine if we have many columns by using `apply`
        df_bq[date_cols] = df_bq[date_cols].apply(pd.to_datetime)
    except Exception as e:
        print("Try to convert `BQ` Datetime gets error: {}".format(e))
        sati = False
    
    try:
        df_json[date_cols] = df_json[date_cols].apply(pd.to_datetime)
    except Exception as e:
        print("Try to convert `JSON` Datetime gets error: {}".format(e))
        sati = False
    
    return df_json, df_bq
#     return compare_dfs_shape(date_df_bq, date_df_json)

# df_json, df_bq, date_cols = convert_with_date_cols()
# print("date convert: ", date_cols)

# Here should add a function that convert float, the reason add this function
# is that for 0.5 with only rounding with return 0, in fact we want 1 to be returned.

def convert_float_round(x, n_keep_digists=4):
    with localcontext() as ctx:
        ctx.rounding = ROUND_HALF_UP
        return float(round(Decimal(x), n_keep_digists))


def convert_percen_cols(df_json, 
                       df_bq, 
                       float_round_estimation = 4,
                       per_threshould = .9):
    """This is a Pipeline, DF's dtype will be same after we have processed.

    Args:
        df_json ([type]): [description]
        df_bq ([type]): [description]
        float_round_estimation ([type], optional): [description]. Defaults to 4.
        per_threshould ([type], optional): [description]. Defaults to .9.

    Returns:
        [type]: [description]
    """
    same_dtype_cols, _ = get_same_diff_type_cols(df_json, df_bq)
    other_not_sati_cols = list(set(list(df_bq.columns)) - set(same_dtype_cols))
    
    if len(other_not_sati_cols) == 0:
        print("There isn't not others type of columns in BigQuery DataFrame.")
        return df_json, df_bq
    
    
    other_sati_json_df = df_json[other_not_sati_cols].astype(str)
    other_sati_bq_df = df_bq[other_not_sati_cols]

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
    
    if len(percen_cols) == 0:
        print("There isn't percentage columns in JSON DataFrame.")
        return df_json, df_bq
    
    print("Get columns: {} as Percentage column to process.".format('\t'.join(percen_cols)))
        
    # If we have get percentage columns, then need to convert them into float
    other_sati_json_df[percen_cols] = other_sati_json_df[percen_cols].applymap(lambda x: float(x.replace('%', ''))/100)
    
    # Key notes here: WE SHOULDN'T COMPARE FLOAT, SHOULD CONVERT INTO STRING!
    # convert BQ df either, so could compare easy...Let's just hard-code this for 4-digits to keep
    per_convert_json = other_sati_json_df[percen_cols].applymap(lambda x: "%.4f" %  round(x, float_round_estimation))
    
    # Here I add with rounding logic that will convert float into rounding logic, but only for BQ DF only!
    other_sati_bq_df[percen_cols] = other_sati_bq_df[percen_cols].applymap(lambda x: convert_float_round(x))
    
    per_convert_bq = other_sati_bq_df[percen_cols].applymap(lambda x: "%.4f" % round(x, float_round_estimation))

    # write these columns back with these new DFs.
    df_json[percen_cols] = per_convert_json
    df_bq[percen_cols] = per_convert_bq
    
    return df_json, df_bq

# df_json, df_bq, percen_cols = convert_percen_cols()

# print("BQ DF dtypes:", df_bq.dtypes)
# print("JSON DF dtypes:", df_json.dtypes)

# remove full special characters that we may face.



def convert_other_spe_cols(df_json, df_bq):
    """As this is a pipeline, after each step process, then we will convert diff columns into same.

    So Here if we need to try with other columns types, then we could just try to set data from original.

    Args:
        df_json ([type]): [description]
        df_bq ([type]): [description]

    Returns:
        [type]: [description]
    """
    same_dtype_cols, _ = get_same_diff_type_cols(df_json, df_bq)
    other_str_cols = list(set(list(df_bq.columns)) - set(same_dtype_cols))
    special_characters = re.escape(string.punctuation)

    def remove_spe_cha(x):
        return re.sub(r"[" + special_characters + "]", "", str(x))

    if not other_str_cols:
        print("There isn't any other String columns to process.")
        return df_json, df_bq
    
    df_bq[other_str_cols] = df_bq[other_str_cols].applymap(lambda x: remove_spe_cha(x))
    df_json[other_str_cols] = df_json[other_str_cols].applymap(lambda x: remove_spe_cha(x))

    return df_json, df_bq


def publish_message(message, topic_id=None, project_id=None):
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_id)
    
    res = publisher.publish(topic_path, message.encode("utf-8"))
    
    try:
        print(res.result())
    except Exception as e:
        print("When try to publish message get error: {}".format(e))



def compare_main_logic(event, context):
    """Main entry point.

    Args:
        event ([type]): [description]
        context ([type]): [description]
    """
    # TODO: Change this for devops.
    project_id = "sbx-65343-autotest8-b-a80bc33f"
    topic_id = "dv_test_data_result"

    # Here we than we could just try to call main compare logic
    message = json.loads(base64.b64decode(event['data']).decode('utf-8'))

    identity = message["id"]
    query = message["query"]
    data = message["data"]
    mode = message["mode"]

    df_bq = get_bq_df(query)
    df_json = get_json_df(data)
    print("Get BQ:",  df_bq.head())
    print("Get JSON: ", df_json.head())
    # Change JSON dataframe columns to same as BQ, so no need for column name failure
    df_json.columns = df_bq.columns

    df_json, df_bq = convert_date_cols(df_json, df_bq)
    df_json, df_bq = convert_percen_cols(df_json, df_bq)
    df_json, df_bq = convert_other_spe_cols(df_json, df_bq)

    status = compare_main(df_json, df_bq)
    print("After full process, get comparation result: ", status)

    message = json.dumps({"id": identity, "status":status, "missed_expected":[], "missed_actual":[]})

    publish_message(message, topic_id, project_id)

    print("Full processing finished :)")