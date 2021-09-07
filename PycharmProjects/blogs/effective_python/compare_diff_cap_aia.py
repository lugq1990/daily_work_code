"""Compare CAP and AIA data is same or not!"""
import os
import logging
import pandas as pd

logger = logging.getLogger("Comparation")


def load_data_log(country, file_path=os.path.abspath(os.curdir), file_sep='---'):
    if not country.startswith(".txt"):
        country += ".txt"

    file_path = os.path.join(file_path, country)

    if not os.path.exists(file_path):
        raise FileNotFoundError("Couldn't file file in folder: {}".format(file_path))

    with open(file_path, 'r') as f:
        data = f.read()

    cap_table, aia_log = data.split(file_sep)

    return cap_table, aia_log


def compare_diff(cap_table, aia_log, save_to_file=True, save_file_path=None, save_file_folder='Compare',
        file_prefix=None, file_name='cap_aia_compare', file_with_date=True, file_date=None):
    """main func to do comparation."""
    cap_dict = {}
    for x in cap_table.split("\n"):
        if x == '':
            continue
        x = x.split("|")
        cap_dict[x[1].strip().replace("_migrate", '')] = x[-2].strip()

    data = [x for x in aia_log.split("\n") if not x.lower().startswith("info") and x != '']

    aia_dict = {}
    for d in data:
        d = d[d.index('Loaded') + 7:]
        d = d.split(" ")
        aia_dict[d[-1].split(".")[-1]] = d[0]

    cap_dict = {k.replace('_dump', '').replace("_delta", ''):v for k, v in cap_dict.items()}
    aia_dict = {k.replace('_dump', '').replace("_tmp", ''):v for k, v in aia_dict.items()}

    print("Get Cap Data number: {}".format(len(cap_dict)))
    print("Get AIA Data number: {}".format(len(aia_dict)))

    combine_dict = {}

    error_num = 0

    for k in cap_dict.keys():
        if k not in aia_dict.keys():
            print("\t Key `{}` not in AIA".format(k))
            combine_dict[k] = [cap_dict[k], 0]
            error_num += 1
            continue
        if aia_dict[k] != cap_dict[k]:
            print("\t Get diff key: {} with AIA: {} and CAP: {}".format(k, aia_dict[k], cap_dict[k]))
            error_num += 1
        combine_dict[k] = [cap_dict[k], aia_dict[k]]
    
    if error_num == 0:
        print()
        print("\tGood news: Test passed!")

    
    if save_to_file:

        if not save_file_path:
            save_file_path = os.path.abspath(os.curdir)

        # first try to make a folder here, so that we don't need to look for the file from other place
        save_file_path = os.path.join(save_file_path, save_file_folder)
        os.makedirs(save_file_path, exist_ok=True)
        
        if file_prefix:
            file_name = file_prefix + "_" + file_name
        
        if file_with_date:
            try:
                if not file_date:
                    file_date = data[-1].split(" ")[0]
                    file_name += "_" + file_date
            except:
                raise ValueError("Couldn't get date automate, please provide `file_date` if still want to combine with date")
    
        if not file_name.startswith("csv"):
            file_name += '.csv'

        df = pd.DataFrame(combine_dict).T
        df.columns = ['CAP', 'AIA']

        df.to_csv(os.path.join(save_file_path, file_name))


if __name__ == '__main__':

    country_list = ['philippines', 'india', 'brazil']
    for country in country_list:
        print("Now is to process country: [{}]".format(country))
        cap_table, aia_log = load_data_log(country)

        compare_diff(cap_table, aia_log, file_prefix=country)

        print("-----Finished process country: [{}]-----".format(country))