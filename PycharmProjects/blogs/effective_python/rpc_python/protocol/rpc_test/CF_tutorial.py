# try to use Python to do CF recommendation

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

user_name_list = ['lu', 'liu', 'baby', 'new_user']
item_list = ['food', 'coffee']
rating = np.random.randn(len(user_name_list), len(item_list))
np.where(rating == 0)

df = pd.DataFrame(rating, columns=item_list, index=user_name_list)

# First is User-based to compute similarity and make recommendation
user_sim = np.zeros((len(user_name_list), len(user_name_list)))

for i in range(len(user_name_list)):
    for j in range(len(user_name_list)):
        user_sim[i, j] = sp.stats.spearmanr(rating[i, :], rating[j, :]).correlation

# plot similarity
sns.heatmap(user_sim, annot=True)

# try to get last user recommendation based on whole users.
user_index = len(rating) - 1
last_user_sim = user_sim[:, user_index]
# let's try to recommend last user for item 1
item_index = 0
weighted_item = np.average(np.sum(last_user_sim * rating[:, item_index]))

