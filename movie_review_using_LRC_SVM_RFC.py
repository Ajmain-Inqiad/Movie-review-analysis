import json
import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

#from time import time 
#from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import make_moons
from sklearn.svm import LinearSVC, SVC
from sklearn.utils import shuffle
from sklearn.preprocessing import Binarizer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_files
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix   

class sentimental_analysis:
    
    def load_files(self, files):
        return load_svmlight_files(files, n_features=None, dtype=None)

    # Calculating Tf-Idf for training and testing
    def tfidf(self, training_data, testing_data):
        tf_transformer = TfidfTransformer()

        print("Training_data TF-IDF")
        #  It computes the TF for each review, the IDF using each review, and finally the TF-IDF for each review
        training_data_tfidf = tf_transformer.fit_transform(training_data)
        print(training_data_tfidf.shape)

        print("Testing_data TF-IDF")
        # .transform on the testing data which computes the TF for each review, 
        # then the TF-IDF for each review using the IDF from the training data 
        testing_data_tfidf = tf_transformer.transform(testing_data)
        print(testing_data_tfidf.shape)

        return [training_data_tfidf,testing_data_tfidf]

    # Binerize target data

    # Converting target into binary
    def binerize (self, raw_target):    
        binerize_target = []
        for i in range(len(raw_target)):
            if raw_target[i] > 5:
                binerize_target.append(1) # Positive
            else:
                binerize_target.append(0) # Negative
        return binerize_target

    # Train and test Logistic Regression Classifier
    def lrc(self, training_data, raw_training_target, testing_data, raw_testing_target):
        print("Binerizing target ...")
        training_target = self.binerize(raw_training_target)
        testing_target = self.binerize(raw_testing_target)
        #start = time()
        logreg = LogisticRegression()
        print("Training ...")
        logreg.fit(training_data, training_target)
        print("Training Done")
        print("Testing ...")
        t1 = logreg.predict(testing_data)
        logreg_accuracy = logreg.score(testing_data, testing_target) * 100
        print("Confusion Matrix for Logistic Regression Classifier: ")
        print(confusion_matrix(testing_target, t1))
        
        #x_train01 = sequence.pad_sequences(x_train, maxlen=400)
        #print(type(x_train))
        #end = time()
        return [logreg, round(logreg_accuracy,2)]
    
    # Train and test Linear SVM Classifier without parameter 
    def lSVC(self, training_data, raw_training_target, testing_data, raw_testing_target, parameter=False):
        print("Binerizing target ...")
        training_target = self.binerize(raw_training_target)
        testing_target = self.binerize(raw_testing_target)
        #start = time()
        clf_linear = LinearSVC()
        print("Training ...")
        clf_linear.fit(training_data, training_target)
        print("Training Done")
        print("Testing ...")
        t1 = clf_linear.predict(testing_data)
        result_lSVC = clf_linear.score(testing_data, testing_target)*100
        print("Confusion Matrix for Linear SVM Classifier: ")
        print(confusion_matrix(testing_target, t1))
        #end = time()
        return [clf_linear, round(result_lSVC,2)]

    # Train and test Random Forest Classifier
    def random_forest(self, training_data, raw_training_target, testing_data, raw_testing_target):
        print("Binerizing target ...")
        training_target = self.binerize(raw_training_target)
        testing_target = self.binerize(raw_testing_target)
        #start = time()
        print("Training ...")
        clf_forest = RandomForestClassifier(n_estimators = 100, min_samples_leaf=5, max_features='auto', max_depth=16)
        clf_forest.fit(training_data, training_target)
        print("Training Done")
        print("Testing ...")
        t1 = clf_forest.predict(testing_data)
        clf_forest_accuracy = clf_forest.score(testing_data, testing_target)*100
        print("Confusion Matrix for Random Forest Classifier: ")
        print(confusion_matrix(testing_target, t1))
        #end = time()
        return [clf_forest, round(clf_forest_accuracy,2)]


# Store path in array for training and testing files
files = ["dataset/train/labeledBow.feat","dataset/test/labeledBow.feat"]

# Object for sentiment_analysis
sa = sentimental_analysis()

# Load data for training_data, training_target and testing_data, testing_target 
print("Loading Files ...")
training_data, raw_training_target, testing_data, raw_testing_target = sa.load_files(files)
print("Done")

# Count tf-idf for training and testing data
tfidf_data = sa.tfidf(training_data, testing_data)

training_data = tfidf_data[0]
testing_data = tfidf_data[1]

print("Logistic Regression Classifier")
result = sa.lrc(training_data, raw_training_target, testing_data, raw_testing_target)
obj_lrc = result[0]
print("Accuracy = ", result[1], "%")

print("Linear SVM Classifier ")
result = sa.lSVC(training_data, raw_training_target, testing_data, raw_testing_target)
obj_lSCV = result[0]
print("Accuracy = ", result[1], "%")

print("Random Forest Classifier")
result = sa.random_forest(training_data, raw_training_target, testing_data, raw_testing_target)
obj_random_forest = result[0]
print("Accuracy = ", result[1], "%")

