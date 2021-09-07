import os
import pandas as pd

path = r"C:\Users\guangqiang.lu\Downloads\remove_duplicate"

file_name = "case_202106-07.xls"
new_file_name = file_name.split(".")[0] + '_remove_duplicate' + '.xlsx'
# This is main key to be represented as unique key
main_key = 'key1'


def remove_duplicate(base_sheet_name=None):

    # Get sheet names
    xl = pd.ExcelFile(os.path.join(path, file_name))

    sheet_names = xl.sheet_names

    # To remove first sheet duplicate
    if not base_sheet_name:
        base_sheet_name = sheet_names[0]
        
    base_df = pd.read_excel(os.path.join(path, file_name), sheet_name=base_sheet_name)
    base_df = base_df.drop_duplicates(subset=main_key)

    # make with datetime column
    base_df.loc[:, base_df.dtypes == 'datetime64[ns]'] = base_df.loc[:, base_df.dtypes == 'datetime64[ns]'].apply(lambda x: pd.to_datetime(x))

    # Write back into a excel
    writer = pd.ExcelWriter(os.path.join(path, new_file_name))
    base_df.to_excel(writer, sheet_name=sheet_names[0], index=False)

    if len(sheet_names) > 1:
        # there are other sheet, read them and just dump them without transformation.
        for name in sheet_names[1:]:
            tmp_df = pd.read_excel(os.path.join(path, file_name), sheet_name=name)
            tmp_df.loc[:, tmp_df.dtypes == 'datetime64[ns]'] = tmp_df.loc[:, tmp_df.dtypes == 'datetime64[ns]'].apply(lambda x: pd.to_datetime(x))
            tmp_df.to_excel(writer, sheet_name=name, index=False)

    writer.close()  

    print("Transformation Finished, please check in folder: {}".format(path))
    

if __name__ == '__main__':
    remove_duplicate()