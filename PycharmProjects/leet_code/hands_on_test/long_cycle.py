# Let's check with classification
import numpy as np
import matplotlib.pyplot as plt

data = np.array([1, 4, 5, 10])
label = np.array([0, 1, 1, 0])

# sort data
index = np.argsort(data)
data = data[index]
label = label[index]

# plot data
plt.scatter(data, label)
plt.legend()

# define gain func
def classification_gain(res, prob, la=0):
    return np.sum(res)**2/(np.sum(prob * (1-prob)) + la)

# get data interval
interval_list = [(data[i]+ data[i+1])/2 for i in range(len(data)-1)]

# compute residual
init_guess = .5
residual = label - init_guess

# init probability
prob = np.array([init_guess] * len(data))
init_gain = classification_gain(residual, prob)

# get cover, if cover is over min_child_split, then remove that leaf
init_cover = 2
init_gamma = 1

def _get_cover(prob, la=0):
    return np.sum(prob * (1 - prob)) - la

# loop for each interval to compute gain
def _get_logic_not(sati):
    return np.logical_not(sati)

gain_list = []
for interval in interval_list:
    left_sati = data <= interval
    left_gain = classification_gain(residual[left_sati], prob[left_sati])
    right_gain = classification_gain(residual[_get_logic_not(left_sati)], prob[_get_logic_not(left_sati)])

    if left_gain < init_gamma:
        print("Shouldn't split to left")
    if right_gain < init_gamma:
        print("Shouldn't spilt to right")
    
    gain_list.append(left_gain + right_gain - init_gain)

# get max gain with  interval list to get split data
split_data= interval_list[np.argmax(gain_list)]

# print tree
left_sati = data <= split_data
print("Build tree")
print(data)
print(data[left_sati], data[_get_logic_not(left_sati)], sep='\t')
print("Tree Finished")


# computer left cover and right to decide to prune or not
left_cover = _get_cover(prob[left_sati])
right_cover =_get_cover(prob[_get_logic_not(left_sati)])
print("Left cover is over init?", left_cover > init_cover)
print("Right cover is over init?", right_cover > init_cover)


# get output with log(odds)
def log_odds(x):
    return np.log(x / (1-x))

def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

# compute log_odds list
odds_list =[]
for d in data:
    left_sati = d < split_data
    if left_sati:
        out = classification_gain(residual[left_sati], prob[left_sati])
    else:
        out = classification_gain(residual[_get_logic_not(left_sati)], prob[_get_logic_not(left_sati)])
    log_out = log_odds(out)
    odds_list.append(log_out)

# set learning_rate and compute probability
learning_rate = .3
prob = [init_guess + learning_rate * sigmoid(odds) for odds in odds_list]

# plot original and prediction
plt.scatter(data, label, label='original')
plt.scatter(data, prob, label='probalitity')
plt.legend()

plt.show()



from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from dtreeviz.trees import dtreeviz

x, y = load_iris(return_X_y=True)
dtc = DecisionTreeClassifier()
dtc.fit(x, y)
viz = dtreeviz(dtc, x, y, target_name='iris', feature_names=list('abcd'), orientation='LR', fancy=False)
# viz.view()
path = r"C:\Users\guangqiang.lu\Documents\lugq\github_code\daily_work_code\PycharmProjects\leet_code\hands_on_test"
viz.save(path)

# explain
# print(explain_prediction_path(dtc, x[0], feature_names=list('abcd')), explanation_type='plain_english')

x_new = x[:, :2]
dtc.fit(x_new, y)

from lightgbm import LGBMClassifier
lgb = LGBMClassifier()
lgb.fit(x, y)
