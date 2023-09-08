"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""


# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from utils import *

###############################################################################
# Digits dataset

# 1.read the dataset using read digit functin
X, y = read_digits()




# 4. Data splitting -- to create train, dev and test sets
# Split data into 50% train , 20% dev and 30% text subsets

X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(
    X, y, test_size=0.3,dev_size=0.2
)

X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)

# 5. Model training
# Create a classifier: a support vector classifier
clf=train_model(X_train, y_train, {'gamma':0.001 , "C":1}, model_type="svm")

# # 6. Getting model predictions on test set
# # Predict the value of the digit on the test subset

predict_and_eval(clf,X_test,y_test)
