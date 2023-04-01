
#Source URL: https://stackoverflow.com/questions/46752650/information-gain-calculation-with-scikit-learn

from scipy.stats import entropy
import pandas as pd

import pandas as pd
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import imdb
from tensorflow.keras.layers import LSTM
from keras.layers import Embedding
from keras.preprocessing import sequence

def information_gain(members, split):
    '''
    Measures the reduction in entropy after the split  
    :param v: Pandas Series of the members
    :param split:
    :return:
    '''
    entropy_before = entropy(members.value_counts(normalize=True))
    split.name = 'split'
    members.name = 'members'
    grouped_distrib = members.groupby(split) \
                        .value_counts(normalize=True) \
                        .reset_index(name='count') \
                        .pivot_table(index='split', columns='members', values='count').fillna(0) 
    entropy_after = entropy(grouped_distrib, axis=1)
    entropy_after *= split.value_counts(sort=False, normalize=True)
    return entropy_before - entropy_after.sum()



filename = 'final_experiment.csv'
data = pd.read_csv(filename)
df = data.values
x = df[:, 0:-1]
y = df[:, -1]


#print(len(df[:,0]))


#members = pd.Series(['yellow','yellow','green','green','blue'])
#split = pd.Series([0,0,1,1,0])
for line in range(42):
    members = pd.Series(df[:,line])
    split = pd.Series(y)
    print (information_gain(members, split))
