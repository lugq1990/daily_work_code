"""Automate download PPT files from website using selenium."""
import os
import zipfile
import shutil

import os
import time
# import patoolib 

from selenium.webdriver import Chrome


files = [x for x in os.listdir() if x.endswith('.zip')]


def init_driver():
    # Donwload driver
    from webdriver_manager.chrome import ChromeDriverManager

    driver = Chrome(ChromeDriverManager().install())

    driver.maximize_window()

    org_handler = driver.window_handles[0]
    
    return driver


# give one link and download file
def download_file(base_link, driver, org_handler):
    link = base_link.get_attribute("href")
        
    # open a new tab
    driver.execute_script('''window.open("{}","_blank");'''.format(link))

    # go to that tab
    # driver.get(link)
    windows_name = driver.window_handles[-1]
    driver.switch_to.window(windows_name)

    get_download_url = driver.find_element_by_link_text("==》点击进入第一PPT素材下载页面《==").get_property("href")

    driver.get(get_download_url)

    # to click real link
    windows_name = driver.window_handles[-1]
    driver.switch_to.window(windows_name)

    t = driver.find_element_by_class_name("downloadlist")

    real_download_link = t.find_element_by_link_text("第一PPT素材下载（网通电信双线）").get_property("href")

    driver.get(real_download_link)

    # close tab and go to original tab
    driver.close()
    driver.switch_to.window(org_handler)

    time.sleep(.5)
    
    
# each kind of ppt to download
def download_one_kind(url_links):
    driver= init_driver()
    
    for url in url_links:
        try:
            driver.get(url)
        except:
            print("End of the pages.")
            break

        # get full donwload links
        result = driver.find_element_by_class_name("tplist")

        # get all links!
        r = result.find_elements_by_link_text("下载")
        print("Get {} elements to download!".format(len(r)))

        org_handler = driver.window_handles[0]

        for base_link in r:
            download_file(base_link, driver, org_handler)

    driver.quit()


# how to read chinese file name with zip
import random

def add_random_str_for_same_file(root_path, filename):
    # check target folder files in case there are same files
    ppt_files = [x for x in os.listdir(root_path) if x.endswith('.pptx')]

    if filename in ppt_files:
        filename_split = filename.split(".")

        random_str = str(random.randint(0, 1000))
        filename_split.insert(-2, random_str +".")
        filename = ''.join(filename_split)
    
    return filename


def extract_chi_zip_file(file_path):
    f = zipfile.ZipFile(file_path, 'r')
    
    for fileinfo in f.infolist():
        filename = fileinfo.filename.encode('cp437').decode('gbk')
        if filename.endswith(".html") or filename.endswith('url'):
            continue
            
        if '/' in filename:
            filename = filename.split("/")[-1]
        
        # Should make it as root path
        root_path = os.path.dirname(file_path)
        
        filename = add_random_str_for_same_file(root_path, filename)
        
        outputfile = open(os.path.join(root_path,  filename), "wb")

        shutil.copyfileobj(f.open(fileinfo.filename), outputfile)
        outputfile.close()
        
    f.close()
    
    
    
def extract_rar_file(file_path):    
    # Should make it as root path
    root_path = os.path.dirname(file_path)
    
    util_path = r"D:\software\unrar\UnRAR.exe"

    patoolib.extract_archive(archive=file_path, verbosity=0 ,outdir=root_path, program=util_path)
    
    
    
# copy download folder files into current folder

def move_download_to_target_folder(org_path, target_folder, keep_file_type='pptx'):
    org_path = r"C:\Users\gqian\Downloads"

    files = [x for x in os.listdir(org_path) if x.endswith(".zip")]

    for f in files:
        shutil.move(os.path.join(org_path, f), os.path.join(target_folder, f))

    zip_files = [x for x in os.listdir(target_folder) if x.endswith(".zip")]
    print("there are :{} zip files".format(len(zip_files)))
    
    for f in zip_files:
        try:
            extract_chi_zip_file(os.path.join(target_folder, f))
        except:
            continue
            
    rar_file = [x for x in os.listdir(target_folder) if x.endswith(".rar")]
    print("there are :{} rar files".format(len(zip_files)))
    
    for f in rar_file:
        try:
            extract_rar_file(os.path.join(target_folder, f))
        except:
            continue
        
    
def remove_useless_file(target_folder, keep_file_type='pptx'):
    # remove useless files
#     if keep_file_type != 'pptx':
#         keep_file_type = [keep_file_type, 'pptx']
#     else:
#         keep_file_type  = ['pptx']
#     files = os.listdir(target_folder)
#     for f in files:
#         if os.path.isfile(os.path.join(target_folder, f), f):
#             extension = f.split(".")[-1]
#             if extension not in keep_file_type:
#                 os.remove(os.path.join(target_folder, f))
    files = [x for x in os.listdir(target_folder) if x.endswith('.zip') or x.endswith('.rar')]
    
    for f in files:
        try:
            os.remove(os.path.join(target_folder, f))
        except:
            continue
        
        

def pipeline_download(url, second_url, target_folder, start_page=2, end_page=5):
    print("Pipeline started!")
    
    os.makedirs(target_folder, exist_ok=True)
    
    url_links = [second_url.format(i) for i in range(start_page, end_page)]
    if start_page == 2:
        url_links.insert(0, url)

    # real donwload happens here!
    try:
        download_one_kind(url_links)
    except Exception as e:
        print("When try to download get error: {}".format(e))
    
    print("End of pipeline!")
    
    
    
if __name__ == '__main__':
    # 图标
    url = "https://www.1ppt.com/sucai/tubiao/"
    second_url ="https://www.1ppt.com/sucai/tubiao/ppt_tubiao_{}.html"

    target_folder = "图标"
        
    start=2
    end=6

    pipeline_download(url,  second_url, target_folder, start_page=start, end_page=end)
    
    
    # PPT模板
    url = "https://www.1ppt.com/moban/keji/"
    second_url ="https://www.1ppt.com/moban/keji/ppt_keji_{}.html"
    
    target_folder = "模板"
        
    start=2
    end=20

    pipeline_download(url,  second_url, target_folder, start_page=start, end_page=end)
    
    
    