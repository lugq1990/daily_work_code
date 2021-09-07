import os
import datetime
import shutil
import datetime


base_path = "/sftp/cio.dsd/"
target_path = "/home/ngap.app.dsd/migration"

not_use_folder_pattern = ['temp', 'backup', 'test', 'brazil', 'tamp']

# country_folder_list = ['india', 'brazil', 'philippines']
country_folder_list= ['brazil']

# first let's try to create date folder
current_date = datetime.datetime.now().strftime("%Y-%m-%d")
date_folder = os.path.join(target_path, current_date)
try:
    os.makedirs(date_folder)
except:
    pass


for country in country_folder_list:
    country_path = os.path.join(base_path, country)
    folder_list = [x for x in os.listdir(country_path) 
        if os.path.isdir(os.path.join(country_path, x))]
    print("folder_list:", folder_list)
    # get statisfy folder
    for folder in folder_list:
        if any([folder.lower().find(x) != -1 for x in not_use_folder_pattern]):
            folder_list.remove(folder)
    print("new_folder: ", folder_list)    
    for folder in folder_list:
        folder_path = os.path.join(country_path, folder)
        try:
            os.makedirs(os.path.join(date_folder, country)
        except :
            pass
        folder_target_path = os.path.join(date_folder, country, folder)
        copy_command = 'cp -r %s %s' % (folder_path, folder_target_path)
        try:
            os.system(copy_command)
            print("Folder: %s has been copyed"% folder)
        except IOError as e:
            print("When try to copy folder: %s get error:%s" % (folder, e))
    
