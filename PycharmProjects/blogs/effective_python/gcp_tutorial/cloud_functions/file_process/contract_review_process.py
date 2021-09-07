import os
import pandas as pd


path = r"C:\Users\guangqiiang.lu\Downloads\step_in_evaluate"

files = [x for x in os.listdir(path) if x.endswith('xlsx')]

# print("Get files: ", files)


def get_review_result(file):
    df = pd.read_excel(file, sheet_name="Exported data")

    df = df.drop(df.columns[0], axis=1)

    sati_cols = [i % 2 != 0 for i in range(df.shape[1])]

    # Loop each column and get ratio
    each_contract_res = []
    final_res = []

    tmp_df = df.loc[:, sati_cols]

    for i in range(tmp_df.shape[1]):
        tmp_str = tmp_df.iloc[:, i].dropna().values.tolist()
        yes_list = ['Y' in x for x in tmp_str]
        if len(yes_list) == 0:
#             print("Error to find Y")
            continue
            
        each_contract_res.append(sum(yes_list) / len(yes_list))
        final_res.extend(yes_list)
    if len(final_res) == 0:
        print("For file: {}: There isn't any Y here, is that possible not reviewed?".format(file.split("_")[0]))
        return 
    
    final_score = sum(final_res) / len(final_res) * 100
    final_score = round(final_score, 6) 
    print("File: {} with Final score: {}%".format(file, final_score))
    
    
for file in files:
    get_review_result(file)