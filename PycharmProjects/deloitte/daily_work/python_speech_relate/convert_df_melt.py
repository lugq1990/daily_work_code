import os
import pandas as pd


path = r"/Users/guangqianglu/Downloads"
file_name = "skill.xlsx"

df = pd.read_excel(os.path.join(path,file_name), header=2)

root_col = "名前(CN)"

first_cols = ["OTC", "PTP", "RTR","F&A"]
second_cols = ["マスタガバナンスG", "企画管理部","教育展開G", "税務部","連結決算G"]

df_first = df[[root_col] + first_cols]
df_second = df[[root_col] + second_cols]

# only change these columns
new_cols = [root_col, "data_type", "value"]

melt_df_first = df_first.melt(id_vars=[root_col], value_vars=first_cols).sort_values([root_col])
melt_df_first.columns = new_cols

melt_df_sencond = df_second.melt(id_vars=[root_col], value_vars=second_cols).sort_values([root_col])
melt_df_sencond.columns = new_cols

writer = pd.ExcelWriter(os.path.join(path, 'melt_data.xlsx'), engine = 'xlsxwriter')
melt_df_first.to_excel(writer, sheet_name = 'first', index=False)
melt_df_sencond.to_excel(writer, sheet_name = 'second', index=False)
writer.save()
writer.close()


