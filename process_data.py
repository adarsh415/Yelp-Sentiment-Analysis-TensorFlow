#Build words
# create vacob
# create integer index





import tensorflow as tf
import numpy as np
from collections import Counter

from numpy.random import random

import data_util


file_path='document.csv'

def build_words(filepath):
    df_frame=data_util.create_clean_dataframe(filepath)
    text_df=df_frame['text']
    words=text_df.apply(lambda x : tf.compat.as_str(x).split())
    voca=[]
    for row in words:
        voca.extend(row)
    return voca

def build_vocab(words,vocabulary_size):

    dictionary=dict()
    count = [['UNK',-1]]
    count.extend(Counter(words).most_common(vocabulary_size-1))
    with open('vocab_set.tsv','w') as f:
        for word, _ in count:
            dictionary[word] = len(dictionary)
            f.write(word+"\n")

    reverse_dictionary=dict(zip(dictionary.values(),dictionary.keys()))
    return dictionary,reverse_dictionary

def convert_word_to_index(words,dictionary):
    """ replace each word in data set with its index in dictionary """

    return [dictionary[word] if word in dictionary else 0 for word in words]


def generate_sample(index_words,context_window_size):
    """ Form training pairs according to the skip-gram model. """

    for index , center in enumerate(index_words):
        context=random.randint(1,context_window_size)

        # get a random target before the center word
        for target in index_words[max(0,index-context):index]:
            yield center,target

        # get a random target after the center word
        for target in index_words[index+1:index+context+1]:
            yield center,target

def genrate_batch(iterator,batch_size):
    """ Group a numerical stream into batches and yield them as Numpy arrays. """
    while True:
        center_batch=np.zeros(batch_size,dtype=np.int32)
        target_batch=np.zeros([batch_size,1])

        for index in range(batch_size):
            center_batch[index],target_batch[index]=next(iterator)

        yield center_batch,target_batch



def process_data(vocab_size,batch_size,skip_window):
    vocabulary = build_words(file_path)
    dictionary,_=build_vocab(vocabulary,vocabulary_size)
    index_words=convert_word_to_index(vocabulary,dictionary)
    del vocabulary
    single_gen=generate_sample(index_words,skip_window)
    return genrate_batch(single_gen,batch_size)



process_data(50000,8,2)







