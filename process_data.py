# Build words
# create vacob
# create integer index





import tensorflow as tf
import numpy as np
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from sklearn.preprocessing import OneHotEncoder

import os
import shutil

import random
import string

import data_util
from nltk import word_tokenize


file_path='document.csv'
VOCAB_SIZE=50000


def build_words(filepath):
    df_frame=data_util.create_clean_dataframe(filepath)
    #data_util.create_train_test_csv(filepath)
    data_util.create_complete_file(filepath)
    text_df=df_frame['text']
    words=text_df.apply(lambda x : word_tokenize(str(x)))
    voca=[]
    for row in words:
        voca.extend(row)
    return voca,df_frame


def build_vocab(words,vocabulary_size):

    dictionary=dict()
    count = [['UNK',-1]]
    count.extend(Counter(words).most_common(vocabulary_size-1))
    with open('vocab_set.csv','w') as f:
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

###
#uncomment below lines to generate batch to train word2vec network
###

#def process_data(vocab_size,batch_size,skip_window):
    #words_token,_ = build_words(file_path)
    #voca=clean_data(words_token)
    #dictionary,_=build_vocab(voca,vocab_size)
    #index_words=convert_word_to_index(words_token,dictionary)
    #print (vocabulary)
    #del words_token
    #single_gen=generate_sample(index_words,skip_window)
    #return genrate_batch(single_gen,batch_size)


def create_dictionary(filepath,vocab_size):
    words_token,df_data=build_words(filepath)
    return build_vocab(words_token,vocab_size )

def create_train_input_csv(filepath):
    words,df_text=build_words(filepath)
    dictionary,_=build_vocab(words,VOCAB_SIZE)
    df_text.loc[:,'text']=df_text.loc[:,'text'].apply(lambda x : convert_word_to_index(x.split(),dictionary))

    data_util.create_train_test_csv_from_dataframe(df_text)








#process_data(50000,8,2)

#create_train_test_data(file_path)


#create_dictionary(file_path,VOCAB_SIZE)

create_train_input_csv('yelp.csv')


