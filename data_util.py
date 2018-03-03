#This util is helper to create cleaned csv file
#so that tensorflow TextLineReader class can process it
# Since TextLineReader uses newline as delimiter, this class was not able
# to process multiline review text from yelp data set with multiple newlines
# To make csv compatible to TensorFlow TextLineReader , this util has been created



import pandas as pd
from nltk.corpus import stopwords
import string
from nltk import word_tokenize
import matplotlib.pyplot as plt


col_keep=['stars','text']

def create_dataFrame(filepath):
    # Method to create pandas DataFrame
    # Argument:
    # filepath - absolute path of csv file
    # return:
    # pandas DataFrame

    temp=pd.read_csv(filepath,usecols=col_keep)
    return temp


def clean_text(column_text):
    # Method to cleanup return character and new line character '\n'.
    # Argument:
    # column_text - row text.
    # return:
    # cleaned text data.
    cleaned_words=column_text.split()
    cleaned_words = [w for w in cleaned_words if w not in string.punctuation]
    cleaned_words = [w for w in cleaned_words if w not in stopwords.words('english')]
    cleaned_words = [w for w in cleaned_words if w.isalpha()]


    return " ".join(cleaned_words)


def create_clean_dataframe(filepath):
    # Method to create clean dataframe from original dataframe.
    # Argument:
    # filepath - absolute path of csv file.
    # return:
    # cleaned pandas DataFrame with stars value 1 and 5.

    temp=create_dataFrame(filepath)
    temp=temp.copy()
    temp.loc[:,'text']=temp.loc[:,'text'].apply(lambda x : clean_text(x))
    temp=temp[(temp.stars==5) | (temp.stars==1)]
    return temp

def create_train_test_csv_from_csv(filepath):
    # Method to create clean dataframe from original dataframe.
    # Argument:
    # filepath - absolute path of csv file.
    #
    # created 2 csv files namely train.csv and test.csv of equal size.

    clean_df=create_clean_dataframe(filepath)
    df_size=int(clean_df.shape[0]*(2/3))
    train=clean_df.iloc[0:df_size,:]
    test=clean_df.iloc[df_size:,:]
    train.to_csv('train.csv',index=False)
    test.to_csv('test.csv',index=False)


def create_complete_file(filepath):
    # Method to create clean dataframe from original dataframe.
    # Argument:
    # filepath - absolute path of csv file.
    # return:
    # cleaned pandas DataFrame with stars value 1 and 5.

    temp = create_dataFrame(filepath)
    temp = temp.copy()
    temp.loc[:, 'text'] = temp.loc[:, 'text'].apply(lambda x: clean_text(x))
    temp.to_csv('document.csv',index=False)

def create_train_test_csv_from_dataframe(dataframe):
    # Method to create clean dataframe from original dataframe.
    # Argument:
    # filepath - absolute path of csv file.
    #
    # created 2 csv files namely train.csv and test.csv of equal size.

    clean_df=dataframe
    df_size=int(clean_df.shape[0]*(2/3))
    train=clean_df.iloc[0:df_size,:]
    test=clean_df.iloc[df_size:,:]
    train.to_csv('train.csv',index=False)
    test.to_csv('test.csv',index=False)

if __name__ == '__main__':
    pass
    #dt=create_dataFrame('yelp.csv')
    #print (dt.shape)
    #create_train_test_csv('yelp.csv')
    #create_complete_file('yelp.csv')