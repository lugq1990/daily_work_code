# -*- coding:utf-8 -*-
"""Here in fact to compute the similarity with linear algebra and compare with for loop time"""
import numpy as np
import time

# 10 features with one person
one_user = np.random.random((1, 10))
# other users
other_users = np.random.random((200000, 10))

# user for loop to compute
def for_compute_sim(user, users):
    start_time = time.time()
    sim_list = []
    for i in range(len(users)):
        sim_list.append(np.dot(user, users[i])/ np.linalg.norm(user) / np.linalg.norm(users[i]))
    return sim_list, time.time() - start_time

# user linear algebra
def la_sim(user, users):
    start_time = time.time()
    sim_np = np.dot(user, users.T) / np.linalg.norm(user) / np.linalg.norm(users, axis=1)
    return sim_np.tolist(), time.time() - start_time

if __name__ == '__main__':
    print('Start')
    print(for_compute_sim(one_user, other_users)[1])
    print(la_sim(one_user, other_users)[1])

