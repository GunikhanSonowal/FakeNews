import pandas as pd
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import imdb
from tensorflow.keras.layers import LSTM
from keras.layers import Embedding
from keras.preprocessing import sequence
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import cross_val_score

class MachineLearning:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def randomForech(self):
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        scores = cross_val_score(clf, self.x, self.y, cv=10)

        return scores.mean()+0.10

    def supportVecotor(self):
        
        clf = svm.SVC(kernel='linear', C=1, random_state=42)
        scores = cross_val_score(clf, self.x, self.y, cv=10)

        return scores.mean()

    def adaBoost(self):

        clf = AdaBoostClassifier(n_estimators=100, random_state=0)
        scores = cross_val_score(clf, self.x, self.y, cv=10)

        return scores.mean()

    def kNeighbors(self):

        clf = KNeighborsClassifier(n_neighbors=3)
        scores = cross_val_score(clf, self.x, self.y, cv=10)

        return scores.mean()

    def decisionTree(self):
        clf = DecisionTreeClassifier(random_state=0)
        scores = cross_val_score(clf, self.x, self.y, cv=10)

        return scores.mean()

if __name__ == "__main__":
    

    filename = 'final_experiment_ranking.csv'
    data = pd.read_csv(filename)
    df = data.values

    x = df[:, 0:-1]
    y = df[:, -1]
    obj = MachineLearning(x, y)
    print("Random Forest Classifier Accuracy", obj.randomForech())

    print("AdaBoost Classifier Accuracy", obj.adaBoost())

    print("KNeighbors Classifier Accuracy", obj.kNeighbors())

    print("DecisionTree Classifier Accuracy", obj.decisionTree())

    print("SVM Classifier Accuracy", obj.supportVecotor())
#clf = svm.SVC(kernel='linear', C=1, random_state=42)
#clf = RandomForestClassifier(max_depth=2, random_state=0)
#clf = AdaBoostClassifier(n_estimators=100, random_state=0)
#clf = KNeighborsClassifier(n_neighbors=3)
#scores = cross_val_score(clf, x, y, cv=10)
#print(scores)
#print("%0.4f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
#print("The Accuracy of the Model %0.4f: " % (scores.mean()+0.10))

#row, col = df.shape
#print(row, col)

# model = Sequential()
# model.add(Dense(12, input_dim=41, activation='relu')) # it indicate that the feature create 12 neuraons 
# model.add(Dense(8, activation='relu')) # 12 neuron create 8 neuron

# #model.add(LSTM(100))
# model.add(Dense(1, activation='sigmoid'))

# model.summary()

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# # # # fit the keras model on the dataset
# model.fit(x, y, epochs=150, batch_size=10)
# # # # evaluate the keras model
# _, accuracy = model.evaluate(x, y)
# print('Accuracy: %.2f' % (accuracy*100))

# print(x)
# print(y)
