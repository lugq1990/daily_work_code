# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import datetime
import os

path = 'C:/Users/guangqiiang.lu/Documents/lugq/workings/201903/hp'
org_file = 'org.xlsx'
add_file = 'add.xlsx'

org_df = pd.read_excel(os.path.join(path, org_file), converters={'date':pd.to_datetime})
add_df = pd.read_excel(os.path.join(path, add_file), converters={'End Time':pd.to_datetime})

# Here is to make date as week day or weekend
week_list = []
for i in range(len(org_df)):
    t = org_df.date.astype(object)[i].isoweekday()
    week_list.append(1 if t < 6 else 0)
week_df = pd.DataFrame(week_list)
cols = org_df.columns.tolist()

org_df = pd.concat([org_df, week_df], axis=1)
cols.append('week')
org_df.columns = cols

### According to week day and weekend to give different default value
t1 = org_df[org_df.week == 1]
t2 = org_df[org_df.week == 0]

t1.fillna(8.0, inplace=True)
t2.fillna(0.0, inplace=True)

org_df = pd.concat([t1, t2], axis=0)


# Just to get what columns that I need
add_df['date'] = add_df['End Time'].apply(lambda x: str(x)[:10])
add_df['id'] = add_df['Employee'].apply(lambda x: x.split('-')[0])
add_df['type'] = add_df['Kind'].apply(lambda x: 1 if x== 'Shift+' else 0)

sec_df = add_df[['Duration', 'id', 'type', 'date']]
sec_df.columns = ['time', 'id', 'type', 'date']


# Here is to add the new date for later use case
new_time_list = []
for i in range(len(sec_df)):
    t = sec_df.time.iloc[i]
    if sec_df.type.iloc[i] == 1:
        new_time_list.append(t)
    else:
        new_time_list.append(-t)


sec_df['new_time'] = pd.DataFrame(new_time_list)

# according to different type to give different signature as pos and neg
tmp = org_df
sec_df.id = sec_df.id.astype(np.int64)

sec_df = sec_df.groupby(['id', 'date']).agg({'new_time':sum}).reset_index()

tmp.date = tmp.date.astype(str)


# Convert raw based dataframe to columns based dataframe
### This is used to convert original file to be a three columns table
tmp.columns = [str(x) for x in tmp.columns]

cols = tmp.columns

re = tmp[cols[0:2]]
re['id'] = cols[1]
re.columns = ['date', 'value', 'id']

date = pd.DataFrame(re['date'])

for i in range(len(cols) - 1):
    if i == 0: continue
    t = tmp[['date', cols[i]]]
    t['id'] = cols[i]
    t.columns = ['date', 'value', 'id']
    re = pd.concat([re, t], axis=0)


# Here I want to use spark to make the DataFrame to table to use SQL
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

re.id = re.id.astype(str)
sec_df.id = sec_df.id.astype(str)

# Table 1 is transferred dataframe
org_spark_df = spark.createDataFrame(re)
# Table 2 is time table after transferred
add_spark_df = spark.createDataFrame(sec_df)

# Now create tables using Spark DataFrame
org_spark_df.registerTempTable('org')
add_spark_df.registerTempTable('add')


# Here is main SQL for getting result.
result_spark_df = spark.sql("""
select id, date, if(new_time is NULL, value, value+new_time) as time 
from
(select t1.id, t1.date, t1.value, t2.new_time
from org t1 
left join add t2
on t1.id = t2.id and t1.date = t2.date)m order by id, date
""")

res_df = result_spark_df.toPandas()
res_df.id = res_df.id.astype(str)


# Here according to the res_df to make the final result DataFrame
# This is ID list
id_list = [str(x) for x in set(res_df.id)]

result_output = res_df[res_df.id == id_list[0]][['date', 'time']]
result_output.columns = ['date', id_list[0]]

result_output.reset_index()
# Loop for the id_list with res_df
for i in range(len(id_list)):
    if i == 0: continue
    length_id = len(res_df[res_df.id == id_list[i]]['time'])
    if length_id / 31 > 1:
        tm = res_df[res_df.id == id_list[i]]['time'][np.arange(1, length_id + 1) % 2 == 1]
        # result_output[id_list[i]] = tm
        result_output[id_list[i]] = list(tm)
    else:
        result_output[id_list[i]] = list(res_df[res_df.id == id_list[i]]['time'])



# Here I have finished all logic, so get original dataframe that should make and convert the columns
name_excel = 'name_list.xlsx'
columns_df = np.array(pd.read_excel(os.path.join(path, name_excel)))

employee_dict = {str(x[1]):x[0] for x in columns_df}
# employee_list = list(columns_df.Employee)
# employee_dict = {x.split('-')[0]: x.split('-')[1] for x in employee_list}

columns_list = [str(x) for x in result_output.columns]
employ_key = employee_dict.keys()

for i in range(len(columns_list)):
    if columns_list[i] == 'date':
        continue
    if columns_list[i] not in employ_key:
        continue
    columns_list[i] = str(columns_list[i] +"-"+ employee_dict[str(columns_list[i])])

result_output.columns = columns_list
print('Final result:')
result_output.head()

### Here I have finished all step, save result to disk
output_file_path = os.path.join(path, 'final_output.xlsx')
if os.path.exists(output_file_path):
    os.remove(output_file_path)
result_output.to_excel(output_file_path, index=False)