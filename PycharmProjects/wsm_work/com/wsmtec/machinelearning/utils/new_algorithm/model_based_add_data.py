# -*- coding:utf-8 -*-
import numpy as np
from sklearn.externals import joblib

def recusive_loop(estimator,trained_data,trained_label,
                  test_data,test_label,
                  thre=.9,pos_ratio=1,iters=15,max_nums=50000,save_path=None):
    data_min_max = trained_data
    label_np = trained_label
    test_data_min_max = test_data
    test_label_np = test_label
    lr = estimator
    miss_return_data = np.empty((1, data_min_max.shape[1]))
    miss_return_label = np.empty((1))
    sati_return_data = np.empty((1, data_min_max.shape[1]))
    sati_return_label = np.empty((1))
    path = save_path
    i = 0
    while (True):
        print("*****************the next iter ****************")
        print("start train the model step %d"%(i+1))
        print("the data shape is ", data_min_max.shape)
        # build the model
        lr.fit(data_min_max, label_np)
        # predict the prob and pred for the test data
        pred = lr.predict(test_data_min_max)
        prob = lr.predict_proba(test_data_min_max)

        # get the correct and miss pred index
        correct_pred = (test_label_np == pred)
        miss_pred = (test_label_np != pred)
        # get the correct data prob(for the next step to depart the threshold for correct pred data) and miss data
        correct_prob = prob[correct_pred]
        correct_data = test_data_min_max[correct_pred]
        correct_label = test_label_np[correct_pred]
        # get the correct pred prob for pos and neg data depart
        correct_pos_prob = correct_prob[correct_label == 1][:,1]
        correct_neg_prob = correct_prob[correct_label == 0][:,0]
        correct_pos_data = correct_data[correct_label == 1]
        correct_neg_data = correct_data[correct_label == 0]
        correct_pos_label = correct_label[correct_label == 1]
        correct_neg_label = correct_label[correct_label == 0]
        # get the satisified data and not satisified data by the threshold
        sati_pos_data = correct_pos_data[correct_pos_prob > thre]
        sati_neg_data = correct_neg_data[correct_neg_prob > thre]
        not_sati_pos_data = correct_pos_data[correct_pos_prob <= thre]
        not_sati_neg_data = correct_neg_data[correct_neg_prob <= thre]
        # get the satisified ant not sati label
        sati_pos_label = correct_pos_label[correct_pos_prob > thre]
        sati_neg_label = correct_neg_label[correct_neg_prob > thre]
        not_sati_pos_label = correct_pos_label[correct_pos_prob <= thre]
        not_sati_neg_label = correct_neg_label[correct_neg_prob <= thre]
        # get the miss pred data
        miss_data = test_data_min_max[miss_pred]
        miss_label = test_label_np[miss_pred]

        # now combine the all sati and not sati data to be one dataset
        # all_sati_data = np.concatenate((data_min_max, sati_pos_data, sati_neg_data), axis=0)
        # all_sati_label = np.concatenate((label_np, sati_pos_label, sati_neg_label), axis=0)
        ##change the sati data(not included the pos data because pos data is so many,
        ##just not want to change the original data prob distribution )

        ####for now I will random add the neg data(that is the 0) to the dataset
        sati_neg_data,sati_neg_label = shuffle(sati_neg_data,sati_neg_label)
        random_add_data = sati_neg_data[:sati_pos_data.shape[0]*pos_ratio,:]
        random_add_label = sati_neg_label[:sati_pos_data.shape[0]*pos_ratio]
        #the neg data leaved
        sati_neg_data = sati_neg_data[sati_pos_data.shape[0]*pos_ratio:,:]
        sati_neg_label = sati_neg_label[sati_pos_data.shape[0]*pos_ratio:]
        #now get the all pos data and added random_add_data
        all_sati_data = np.concatenate((data_min_max, sati_pos_data,random_add_data), axis=0)
        all_sati_label = np.concatenate((label_np, sati_pos_label,random_add_label), axis=0)

        #return the miss data and label,
        #add the postive data to return(that means training pos data will not changed)
        miss_return_data = np.concatenate((miss_data,not_sati_neg_data, sati_neg_data,not_sati_pos_data), axis=0)
        miss_return_label = np.concatenate((miss_label,not_sati_neg_label,sati_neg_label, not_sati_pos_label), axis=0)

        # now that we get all the satisified and not satisified data and label ,we can train the model again
        # just make a little simplified transformation ,make the train data to be data_min_max
        data_min_max = all_sati_data
        label_np = all_sati_label
        test_data_min_max = miss_return_data
        test_label_np = miss_return_label

        print("finished the step  %d"%(i+1))
        i += 1
        if (i > iters or data_min_max.shape[0] > max_nums):
            #save the model
            joblib.dump(lr,path+'//lr.pkl')
            sati_return_data = np.delete(data_min_max,0,axis=0)
            sati_return_label = np.delete(label_np,0,axis=0)
            miss_return_data = np.delete(miss_return_data,0,axis=0)
            miss_return_label = np.delete(miss_return_label,0,axis=0)
            return sati_return_data,sati_return_label,miss_return_data,miss_return_label
            break