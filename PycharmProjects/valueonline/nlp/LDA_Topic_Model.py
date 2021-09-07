# -*- coding:utf-8 -*-
"""This is just use the LDA and NMF to do topic modeling
    For topic modeling(there are many documents, I can not get the topic for all documents,
    using lda and nmf to get the given topic and given keywords for each topic)
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# this is just for getting the model topic result.
def display(model, feature_names, top):
    for idx, topic in enumerate(model.components_):
        print('Topic %d'%idx)
        print(','.join([feature_names[i] for i in topic.argsort()[::-1][: -top-1: -1]]))

datasets = fetch_20newsgroups(data_home='C:\\Users\\Asus', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=1234)
data = datasets.data

# get how many data from original datasets.
n_features = 1000

# for NMF model, it can use Tfidf_vector to build model
tfidf_vector = TfidfVectorizer(max_df=.95, min_df=2, max_features=n_features, stop_words='english')
tfidf = tfidf_vector.fit_transform(data)
# get tfidf features
tfidf_features = tfidf_vector.get_feature_names()

# for LDA model, it can just use CountVectorizer to build model
tf_vector = CountVectorizer(max_df=.95, min_df=2, max_features=n_features, stop_words='english')
tf  = tf_vector.fit_transform(data)
# get CounterVectorizer features
tf_features = tf_vector.get_feature_names()

# how many topic wanted to get.
n_topics = 20

### start to build NMF and LDA model
nmf = NMF(n_components=n_topics, alpha=.1, l1_ratio=.5, init='nndsvd', random_state=1234).fit(tfidf)
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=5, learning_method='online', learning_offset=50, random_state=1234).fit(tf)

# after training model, get the different model topic result
# how many top words for each topic
top = 10

print('Here is NMF model Topic')
display(nmf, tfidf_features, top)
print('*'*100)
print('Here is LDA model Topic')
display(lda, tf_features, top)

print('All Done!')