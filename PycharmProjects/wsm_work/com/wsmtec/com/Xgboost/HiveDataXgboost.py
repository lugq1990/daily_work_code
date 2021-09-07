# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
import time

class my_over_sampling:
    def __init__(self,samples,num=10,pos_samples=None):
        self.samples = samples
        self.num = num
        self.n_samples,self.n_cols = samples.shape
        self.pos_samples = pos_samples

    def over_sampling(self):
        self.index = 0
        self.result = np.zeros((self.num,self.n_cols))
        self.centers = np.mean(self.samples,axis=0)
        #print("center ",self.centers)
        dis = self.__computeDis()[0]
        for i in range(self.num):
            rotio = np.random.random()
            obj_choose = np.random.randint(self.n_samples)
            #add the guassian mixer
            mu, sigma = 0,0.1
            noise_rotio = np.random.normal(mu,sigma,size=(1,self.n_cols))[0]
            #print("!!",noise_rotio[0])
            self.result[i] = self.samples[obj_choose] + rotio*dis + noise_rotio
        return self.result

    #compute the center for the data
    def __computeDis(self):
        num_sam = self.samples.shape[0]
        #add the pos data center and distance
        if(self.pos_samples is not None):
            print("now in the pos data !")
            num_sam = self.pos_samples.shape[0]
            self.n_cols = self.pos_samples.shape[1]
            self.pos_center = np.mean(self.pos_samples,axis=0)
            dis_pos = np.zeros((1,self.n_cols))
            for j in range(num_sam):
                dif_pos = self.pos_center - self.pos_samples[j]
                dis_pos += dif_pos
            avg_pos_dis = dis_pos/num_sam
            return avg_pos_dis
        else:
        #use the original data center and distance
            dis = np.zeros((1,self.n_cols))
            for i in range(num_sam):
                dif = self.centers - self.samples[i]
                dis += dif
            avg_dis = dis/num_sam
            #print(avg_dis[0])
            return avg_dis


path = 'F:\workingData\\201709\data\Hive'
df = pd.read_csv(path+'/test.txt',sep='\t')

tmp = np.arange(1,94)
col = list()
for i in range(tmp.shape[0]):
    col.append(np.str(tmp[i]))
df.columns = col
df.drop(['1','92','93'],axis=1,inplace=True)

#make the data for pos and neg
pos = df.loc[df['91']==0]
pos_data = np.array(pos.drop('91',axis=1))
pos_label = pos['91']
neg = df.loc[df['91']==1]
neg_data = np.array(neg.drop('91',axis=1))
neg_label = neg['91']
print('start generate data')
start = time.time()

my_over = my_over_sampling(samples=neg_data,num=50000).over_sampling()
my_over_label = np.ones((my_over.shape[0]))
print('finished!')
end = time.time()
print('make the data use ',end-start,'s')
# neg_f_data = np.concatenate((neg_data,my_over),axis=0)
# neg_f_label = np.concatenate((neg_label,my_over_label),axis=0)
#the produced and original data combined
data_neg = np.concatenate((neg_data,my_over),axis=0)
label_neg = np.concatenate((neg_label,my_over_label),axis=0)

data = np.concatenate((data_neg,pos_data),axis=0)
label = np.concatenate((label_neg,pos_label),axis=0)

#added standard data methed for the all data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(data)
data = sc.transform(data)


from sklearn.model_selection import train_test_split
df_data = df.drop(['91'],axis=1)
df_label = df['91']
df_data = np.array(df_data)
df_label = np.array(df_label)
#use the produced data
xtrain,_,ytrain,_ = train_test_split(data,label,test_size=.3)
#not use the produced data
#the df_data is transformed
df_data = sc.transform(df_data)
_,xtest,_,ytest = train_test_split(df_data,df_label,test_size=.3)


train_data = xgb.DMatrix(xtrain,label=ytrain)
test_data = xgb.DMatrix(df_data,label=df_label)
watch_list = [(test_data, 'eval'), (train_data, 'train')]
param = {
'max_depth': 8,
'eta': 0.05,
'silent': 1,
'gamma':0,
'subsample':0.8,
'colsample_bytree' : 0.8,
'alpha':1,
'lambda':1,
'objective': 'binary:logistic',
'min_child_weight': 5,
'eval_metric':['auc','error'],
'evals_result':{},
'learning_rates':0.1
}
print("start train the model")

model = xgb.train(param,train_data,num_boost_round=100,evals=watch_list)
# model.dump_model(path+'/xg.raw.txt')
# model.save_model(path+'/xgboost_5w.model')


#evaluate the model
from sklearn import metrics
prod = model.predict(test_data,ntree_limit=model.best_ntree_limit)
auc = metrics.roc_auc_score(df_label,prod)
fpr,tpr,_ = metrics.roc_curve(df_label,prod)
ks = tpr - fpr
pred = np.array(pd.Series(prod).map(lambda x:1 if x>=.5 else 0))
f1 = metrics.f1_score(df_label,pred)
con = metrics.confusion_matrix(df_label,pred)
recall = metrics.recall_score(df_label,pred)
print("auc=",auc)
print("ks=",ks.max(axis=0))
print('f1 score',f1)
print('recall',recall)
print('confusion matrix',con)
#compute the odds and score
# odds = prod / (1 - prod)
# score = 600 - 166.1 * np.log(odds)
# odds_pd = pd.DataFrame(odds)
# odds_pd.to_csv(path+'/odds.csv')
# score_pd = pd.DataFrame(score)
# score_pd.to_csv(path+'/score.csv')
# prod_pd = pd.DataFrame(prod)
# prod_pd.to_csv(path+'/pred.csv')

#draw the gragh
# import matplotlib.pyplot as plt
# plt.plot(fpr, tpr)
# plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='random')
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve-test')
# plt.legend(loc="lower right")
# plt.show()


#make the data combined the data and label, cast it to be a dataframe to save into the disk
my_over_label = my_over_label.reshape(-1,1)
out = np.concatenate((my_over,my_over_label),axis=1)
re = pd.DataFrame(tmp)
# re.to_csv(path+'/neg_data.csv',index=False,header=False)

