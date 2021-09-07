# -*- coding:utf-8 -*-
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

n_features = 500     # This is how many words wanted to be used
n_topics = 10   # How many topic to be extracted
n_words_show = 10   # How many words to be shown from each topic

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('header', 'quotes', 'footers'))
data = dataset.data

# This function is used to display the topic result
def display(model, feature_names, n_show, topic_name=None):
    # The model components just means that each word assign to different topic probability
    # If want to get different document for which topic it assigned to, use model.transform function
    for topic_id, topic in enumerate(model.components_):
        if topic_name is not None:
            print('Topic: {}'.format(topic_name[topic_id]))
        else: print('Topic: {}'.format(topic_id))

        print(' '.join([feature_names[i] for i in topic.argsort()[:-n_show-1: -1]]))

# Make TFIDF features vector for NMF
tfidf_vector = TfidfVectorizer(min_df=2, max_df=.5, max_features=n_features, stop_words='english')
tfidf = tfidf_vector.fit_transform(data)
tfidf_feature_names = tfidf_vector.get_feature_names()     # Which words to be used for TFIDF for 'n_features' words mean


# Make TF features by using count to represent features
tf_vector = CountVectorizer(min_df=2, max_df=.5, max_features=n_features, stop_words='english')
tf = tf_vector.fit_transform(data)
tf_feature_names = tf_vector.get_feature_names()    # Words name for count feature vector


print('Now start to train model!')
# Start to build NMF model and LDA model
nmf = NMF(n_components=n_topics, alpha=.1, l1_ratio=.5, init='nndsvd')
nmf.fit(tfidf)

lda = LatentDirichletAllocation(n_components=n_topics, max_iter=5, learning_method='online', learning_offset=50., random_state=1)
lda.fit(tf)

# Start to display LDA model and NMF model training result, default topic_name is None
# Because I can't get the prior name of different topic, if we can repeat training result,
# and store the model to use transfer each document with which topic assigned to, set topic_name
print('This is NMF model result:')
display(nmf, tfidf_feature_names, n_words_show)

print('*'*20)
print('This is LDA model result:')
display(lda, tfidf_feature_names, n_words_show)

print('Start inferring:')
# Here I also want to use trained model to infer a document to get which topic it belongs to.
print('For using NMF, document belongs to topic: {}'.format(np.argmax(nmf.transform(tfidf[0]), axis=1)))
print('For using LDA, document belongs to topic: {}'.format(np.argmax(lda.transform(tf[0]), axis=1)))

print('All step finished!')

