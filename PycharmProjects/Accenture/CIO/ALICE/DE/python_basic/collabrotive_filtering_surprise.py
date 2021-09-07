# -*- coding:utf-8 -*-
from surprise import Dataset, evaluate, KNNBasic
import os, io

data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()

sim_options = {'name': 'cosine', 'user_based':False}

knn = KNNBasic(sim_options=sim_options)

# start to train the model
knn.train(trainset)

testset = trainset.build_anti_testset()
predictions = knn.predict(testset)

# make the prediction to be readable
from collections import defaultdict

# make the get top k function
def get_topk(pred, topk=3):
    top_direc = defaultdict(list)
    # loop for the whole set with the prediction
    for uid, iid, true_rate, est, _ in pred:
        top_direc[uid].append([iid, est])

    # get top k with predictions
    for uid, ratings in top_direc.items():
        top_direc[uid] = ratings.sort(key=lambda x:x[1], reverse=True)
    return top_direc

# here is to get the movie name
def get_name_of_movie():
    file_name = (os.path.expanduser('~') + '/.surprise_data/ml-100k/ml-100k/u.item')
    iid_to_name = {}
    with io.open(file_name, 'r', 'ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            iid_to_name[line[0]] = line[1]
    return iid_to_name

# first to get the prediction with top n
topN_recom = get_topk(predictions, topk=4)
# get the movie name dictionary
iid_to_name = get_name_of_movie()

# print the prediction with movie name
for user, user_rating in topN_recom.items():
    print(user, [iid_to_name[num] for (num, _) in user_rating])




