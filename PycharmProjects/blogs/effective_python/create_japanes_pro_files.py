import os
import pandas as pd
import numpy as np
import warnings
import copy

warnings.simplefilter('ignore')

year = '2021'

start, end = "5/1/{}".format(year), "5/31/{}".format(year)


root_path = r"C:\Users\guangqiiang.lu\Downloads\japanese_pro\fuso_2"


# convert a new dataframe
def construct_a_new_df(start, end):
    """This df is just a Dataframe with Date"""
    
    df = pd.DataFrame(pd.date_range(start=start, end=end), columns=['Date'])

    # week day
    df["Weekday"] = df['Date'].dt.weekday + 1

    # is japanese holiday
    import jpholiday

    holiday_of_jp = [x[0]  for year in [2018, 2019, 2020, 2021] for x in jpholiday.year_holidays(year)]
    df["Isholiday"] = df['Date'].isin(holiday_of_jp).map(lambda x: 1 if x is True else 0)

    # is work date
    df["IsWorkday"] = df['Date'].dt.dayofweek.map(lambda x: 0 if x in [5, 6] else 1)

    # first day of week
    df["FirstWorkday"] = df['Date'].dt.dayofweek.map(lambda x: 1 if x == 0 else 0)

    # year, month, day
    df["Year"] = df['Date'].dt.year
    df["Month"] = df['Date'].dt.month
    df["Day"] = df['Date'].dt.day

    # is_weekday
    df["IsWeekend"] = df['Date'].dt.dayofweek.isin([5, 6]).map(lambda x: 1 if x is True else 0)

    # week number
    # as for the 2021, the first date is Friday, so this is not correct! so change it manually!
    df.loc[df['Year'] != 2021, "WeekNum"] = df['Date'].dt.weekofyear

    df.loc[df['Year'] == 2021, "WeekNum"] = df['Date'].dt.weekofyear +1
    df.loc[(df['Year'] == 2021) & (df['WeekNum'] == 54), 'WeekNum'] = 1

    # month of week
    # Noted, with just `(d.day-1) // 7 + 1` will get wrong result!
    import calendar

    def week(dt):
        mth = calendar.monthcalendar(dt.year, dt.month)
        for i, wk in enumerate(mth):
            if dt.day in wk:
                return i + 1
            
    df["month_of_the_week"] = df['Date'].apply(lambda x: week(x))
#     df["month_of_the_week"] = df['Date'].apply(lambda d: (d.day-1) // 7 + 1)

    # month_start_week based on `month_of_the_week`
    df["month_start_week"] = df["month_of_the_week"].map(lambda x: 1 if x == 1 else 0)

    # I have to say that I haven't get a better solution here, what I could think is to get each year, month and get max value of week, if equal to max, then is the last
    # month_end_week
    unique_year = list(set(df['Year']))
    unique_month = list(set(df['Month']))
    
    df["month_end_week"] = 0

    for y in unique_year:
        for m in unique_month:
            each_month_max_week = df.loc[(df['Year'] == y) & (df['Month'] == m)]['month_of_the_week']
            
            if len(each_month_max_week) == 0:
                continue
            else:
                each_month_max_week = np.nanmax(each_month_max_week)
                
            if each_month_max_week < 4:
                # for each month, at least have 4 weeks!
                continue
            df.loc[(df['Year'] == y) & (df['Month'] == m) & (df["month_of_the_week"] == each_month_max_week), 'month_end_week'] = 1
            
    df["actual_vol"] = 0
    
    return df



def create_each_folder_new_file(folder_list, root_path=root_path, inplace=False):
    """
    Keep in mind, if set `inplace=True`, then the original file will be overwriten.
    """
    for i in range(len(folder_list)):
        folder_name = folder_list[i]
        print("Start to process folder: {}".format(folder_name))

        folder_path = os.path.join(root_path, folder_name)
        
        file_name = [x for x in os.listdir(folder_path) if not x.startswith('2021') and x.find("auto") < 0][0]
        file_path = os.path.join(folder_path, file_name)

        df = pd.read_excel(file_path)
        
        # new added DF
        new_df = construct_a_new_df(start, end)
        
        if folder_name == "NONPO":
            tmp_df_list = []
            
            type_list = ['Center Scan', 'SC Scan', 'Posting']
            for i in range(len(type_list)):
                tmp_df = copy.deepcopy(new_df)
                tmp_df['Type'] = type_list[i]
                
                tmp_df_list.append(tmp_df)
            
            new_df = pd.concat(tmp_df_list, axis=0)
        
        # change the order with just original DataFrame
        diff_col = list(set(list(df.columns)).symmetric_difference(list(new_df.columns)))
        
        if len(diff_col) > 0:
            print("Get diff columns: ", diff_col)
            continue
        new_df = new_df[list(df.columns)]

        # combine new created and original DF
        output_df = pd.concat([df, new_df], axis=0)
        
        # Save combined DF into Disk
        if not inplace:
            new_file_name = folder_name + "_auto.xlsx"
        else:
            new_file_name = folder_name

        output_df.to_excel(os.path.join(folder_path, new_file_name), index=False)
        print("New file: {} has been saved into disk!".format(new_file_name))

        print("Original shape: `{}` and new shape: `{}`, added: {} rows!".format(df.shape[0], output_df.shape[0], (output_df.shape[0] - df.shape[0])))
        
        print("**** GOOD NEWS! Finished for :{} ****".format(folder_name))
        print()


if __name__ == "__main__":
    folder_list = [x for x in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, x)) and not x.startswith(".")]
    print("Get folder:", folder_list)

    create_each_folder_new_file(folder_list)
