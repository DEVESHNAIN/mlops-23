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
import itertools

###############################################################################
# Digits dataset

# 1.read the dataset using read digit functin
X, y = read_digits()





# 3. Data splitting into multiple ratios to create train and test sets
test_sizes = [0.1, 0.2, 0.3]
dev_sizes = [0.1, 0.2, 0.3]


# print (itertools.product(test_sizes, dev_sizes))

# Create a list of tuples with all combinations of hyper parameter gamma and C
gamma_values = [0.001, 0.01, 0.1, 1, 10, 100]
C_values = [0.1, 1, 2, 5, 10]

hyper_params={}
hyper_params['gamma'] = gamma_values
hyper_params['C'] = C_values

all_possible_param_combinations = fetch_hyperparameter_combinations(hyper_params)
# print(all_possible_param_combinations)

for test_size, dev_size in itertools.product(test_sizes, dev_sizes):
    print(f"test_size={test_size} dev_size={dev_size} train_size={1 - test_size - dev_size}", end=' ')
    
    X_train, X_dev,X_test, y_train, y_dev, y_test = split_train_dev_test(X, y, test_size=test_size, dev_size=dev_size)
    
    # 4. Data preprocessing
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)
    X_dev = preprocess_data(X_dev)

    # HYPERPARAMETER TUNING


    best_hparams, best_model, best_dev_acc = tune_hparams(X_train, y_train, X_dev, y_dev, all_possible_param_combinations)

    # Train  model on best hyper parameter

    trained_model = train_model(X_train, y_train, best_hparams)
    # calculate test and train accuracy on best hyper parameter trained model
    test_accuracy = predict_and_eval(trained_model, X_test, y_test)
    train_accuracy = predict_and_eval(trained_model, X_train, y_train)

    print(f"dev_acc={best_dev_acc:.2f} test_acc={test_accuracy:.2f} train_acc={train_accuracy:.2f}")
    
    # log the best hyperparameters 
    print(f"Best hyperparameters: gamma={best_hparams['gamma']}, C={best_hparams['C']}\n")