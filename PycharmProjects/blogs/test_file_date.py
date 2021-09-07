from datetime import datetime


file_list = ['AgentConnectionDetail_philippines_2020-12-11-021423.csv',
'AgentConnectionDetail_philippines_2020-12-14-041404.csv',
'AgentConnectionDetail_philippines_2020-12-14-051203.csv',
'AgentConnectionDetail_philippines_2020-12-09-051203.csv']

current_date = datetime.utcnow().strftime('%Y-%m-%d')

contain_date_file = any([True if len(x.split("_")) > 2 else False for x in file_list])
if contain_date_file:
    # filter other date except today
    file_list = [x if x.split("_")[-1].replace(".csv", '')[:10] == current_date else None for x in file_list]
    current_date_file_list = []
    for file in file_list:
        if not file:
            continue
        if file.split("_")[-1].replace(".csv", '')[:10] == current_date:
            current_date_file_list.append(file)
    
    file_list = current_date_file_list
    # convert string into datetime for comparation
    if file_list:
        # we need to get max date file name for the list of files in case there are many file with current date
        try:
            file_date_obj = [datetime.strptime(x.split("_")[-1].replace(".csv", ''), "%Y-%m-%d-%H%M%S") 
                for x in file_list]
                # This will ensure there will be the latest file name
            file_list = [file_list[file_date_obj.index(max(file_date_obj))]]
        except Exception as e:
            print("When try to convert to date time, get error:{}".format(e))   


from sklearn.ensemble import GradientBoostingClassifier

# choose a decision tree for classification, fit and make prediction and evaluate to get probability with error,
# update a new weight for each sample, build a new tree based on the new weights, with data and label. repeat.
from sklearn.ensemble import AdaBoostClassifier


from sklearn.ensemble import RandomForestClassifier