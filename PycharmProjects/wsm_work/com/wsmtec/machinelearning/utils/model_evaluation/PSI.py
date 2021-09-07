# -*- coding:utf-8 -*-
import numpy as np

def compute_psi(org_data,new_data,splits=10,min=300,max=900):
    org_num_list,new_num_list,org_psi_list,new_psi_list,psi_list = [],[],[],[],[]
    org_min,org_max,new_min,new_max = org_data.min(),org_data.max(),new_data.min(),new_data.max()
    splits=splits
    min,max = min,max
    if(splits != None):
        range = int((max-min)/splits)
        start_index = min
        #judge if the data min or max lower or uper the params min and max
        if(org_min< min or org_max>max or new_min<min or new_max>max):
            lower_org_index_count = np.sum(org_data <= start_index)
            lower_new_index_count = np.sum(new_data <= start_index)
            upper_org_index_count = np.sum(org_data > max)
            upper_new_index_count = np.sum(new_data > max)
            org_num_list.append(lower_org_index_count)
            org_num_list.append(upper_org_index_count)
            new_num_list.append(lower_new_index_count)
            new_num_list.append(upper_new_index_count)
        for _ in np.arange(splits):
            index = start_index + range
            org_index_count = np.sum((org_data> start_index) & (org_data <= index))
            new_index_count = np.sum((new_data> start_index) & (new_data <= index))
            org_num_list.append(org_index_count)
            new_num_list.append(new_index_count)
            start_index = index
        #compute the total num of the org and new data
        org_num_total = np.sum(org_num_list)
        new_num_total = np.sum(new_num_list)
        #compute each range psi
        for i in np.arange(len(org_num_list)):
            org_psi_list.append(org_num_list[i]/org_num_total)
            new_psi_list.append(new_num_list[i]/new_num_total)
        last_one_org,last_one_new = org_psi_list[1],new_psi_list[1]
        org_psi_list.pop(1)
        org_psi_list.append(last_one_org)
        new_psi_list.pop(1)
        new_psi_list.append(last_one_new)
        #according the pis fun to compute the psi
        for k in np.arange(len(org_psi_list)-1):
            if(org_psi_list[k]==0.0 or new_psi_list[k]==0.0):
                print("the orginal data or new data during the splits range "+np.str(k)+" have zero object!")
                return None,-1
        for j in np.arange(len(org_psi_list)-1):
            ratio_org = org_psi_list[j]/org_num_total
            ratio_new = new_psi_list[j]/new_num_total
            psi_list_ratio = (ratio_org - ratio_new)*np.log(ratio_org/ratio_new)
            psi_list.append(psi_list_ratio)
        psi = np.sum(psi_list)
    return psi_list,psi

a = np.random.randint(100,1500,size=15000)
b = np.random.randint(100,1000,size=1000)
psi_list,psi = compute_psi(a,b,splits=9,min=200)
print(psi)
print(psi_list)




