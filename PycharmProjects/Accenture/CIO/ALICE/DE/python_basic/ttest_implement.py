# -*- coding:utf-8 -*-
"""This is to implement the t student test also know as t-test, as the scipy has the implement with
scipy.stats.ttest_ind to compute the t and p value, here I will implement the logic to compute the
t and p value and compare the result to the scipy implement.
as t test is to compare two different group whether or not is same with each other, ttest is based on
computing the mean value within two different group.
To compute the t test, the first thing is to make the hypothesis: null hypothesis as the group is same and
alternative hypothesis as they are different.
The t-value represents how much they are different, and p-value represents how much probability it's true as
whether or not this happens with chance, smaller p-value is better, as the thing doesn't happen by chance!"""

import numpy as np
from scipy import stats

a = np.random.random((100, ))
b = np.random.random((100, ))

# this function assume that the data should have same records, but if not, the t-value is not same with each other.
t, p = stats.ttest_ind(a, b)

# first to compute the t-value, according to the function:
# t = (mean(a) - mean(b)) / (np.sqrt(s1**2/len(a) + s2**2/len(b)))   s1**2 = (a - a.mean())**2 / (len(a) - 1)
s1 = np.sum((a - a.mean())**2) / (len(a) - 1)
s2 = np.sum((b - b.mean())**2) / (len(b) - 1)

t_ma = (np.mean(a) - np.mean(b)) / (np.sqrt(s1/len(a) + s2/len(b)))

print('t scipy:', t)
print('t manually:', t_ma)
print("t-value is close ?", np.allclose(t, t_ma))

### p-value means the signification and the degree of freedom according to the two sample distribution records
### with n1 + n2 - 2 to get the stats to compute the p-value
df = 2 * len(a) - 2
p_ma = (1 - stats.t.cdf(t_ma, df=df)) * 2

print("p scipy:", p)
print("p manually:", p_ma)
print("p value is same with each other:", np.allclose(p, p_ma))



from pyspark.sql import SparkSession
spark =SparkSession.builder.getOrCreate()

sc = spark.sparkContext()

sc.textFile()



