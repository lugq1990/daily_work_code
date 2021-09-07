import numpy as np
import os
import pandas as pd
import warnings

warnings.simplefilter("ignore")

file_path = r"C:\Users\guangqiiang.lu\Downloads\japanese_pro"
file_name = os.path.join(file_path, "case_March.xls")

df = pd.read_excel(file_name)


pro_filter = "処理タイプ"
other_filter = "事業部"

key_col = "案件ID"

filter_1 = (df[pro_filter].isin(["開始(同条件あり)", "開始(同条件なし)"])) & (df[other_filter].isin(["SSO広域", "SSO首都圏"]))
filter_2 = (df[pro_filter].isin(["開始(同条件あり)", "開始(同条件なし)"])) & (df[other_filter].isin(["TS"]))
filter_3 = (df[pro_filter].isin(["延長＋条変", "条変のみ"]))

df1 = df.loc[filter_1]
df2 = df.loc[filter_2]
df3 = df.loc[filter_3]

# remove duplidate for df2
df3 = df3.drop_duplicates(keep='first')

# print(df1.head(1))
# print(df2.head(1))
# print(df3.head(1))

merge_df = pd.concat([df1, df2, df3], axis=0)

diff_df = pd.merge(df, merge_df, how='left')

diff_df[key_col].isnull().sum()



def get_diff_key(df1):
    # get start key
    start_key = df1.iloc[0, 0]

    # sort id with ascending order
    order_key = df1.sort_values(by=key_col)
    # filter with start key
    order_key = order_key.loc[order_key[key_col] >= start_key]

    # unique key values
    order_key = order_key[key_col].values

    # get continous values from start to end
    continous_keys = np.arange(start_key, max(order_key) +1)

    diff_key = list(set(continous_keys) - set(order_key))
    print("Get diff key number: ", len(diff_key))

    is_sati = len(set(diff_key) & set(order_key)) == 0
    
    if not is_sati:
        raise RuntimeError("Get  diff key in original DF! Please check!")
 
    return diff_key


# get each dataframe diff key result
df_list = [df1, df2, df3]

diff_df_list = [df]

for d in df_list:
    diff_key_list = get_diff_key(d)
    
    key_df = pd.DataFrame(diff_key_list, columns=['Diff key'])
    
    diff_df_list.append(key_df)

# Now to save the each diff key into different sheet
sheet_name_1= "開始SSO"
sheet_name_2 = "開始TS"
sheet_name_3 = "条変"

sheet_name_list = ['Original', sheet_name_1, sheet_name_2, sheet_name_3]

with pd.ExcelWriter(os.path.join(file_path, "Different_key.xlsx")) as writer:
    for i, d in enumerate(diff_df_list):
        d.to_excel(writer, sheet_name_list[i], index=False)
    writer.save()
