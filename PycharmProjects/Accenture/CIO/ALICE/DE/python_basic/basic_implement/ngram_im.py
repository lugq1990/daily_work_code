# -*- coding:utf-8 -*-
"""This is to implement the ngram logic with pure python code."""


def ngram(data_list, n):
    return zip(*[data_list[i:] for i in range(n)])

def nltk_im(data_list, n):
    """This is just to use the nltk to make the ngram, really easy to use,
    but really slower...."""
    import nltk

    if n == 2:
        return list(nltk.bigrams(data_list))
    elif n == 3:
        return list(nltk.trigrams(data_list))
    else:
        raise NotImplementedError("not implement with nltk... in fact, Just I don't search)")

if __name__ == '__main__':
    data = 'I love the feeling to learn'
    data = data.split(' ')
    print(list(ngram(data, 4)))

    print('2th: ', nltk_im(data, 2))
    print('3th: ', nltk_im(data, 3))





