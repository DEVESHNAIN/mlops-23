"""
IIT J Mlops 23 batch repo
Author: Devesh Nain 
Roll No: M22AIE247

"""


#  Python imports
import matplotlib.pyplot as plt

# Importing the  datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from utils import *

# from skimage.transform import rescale, resize
from joblib import dump, load
from sklearn import tree

import pandas as pd
import argparse

###############################################################################
# passing inputs through argument parser

parser = argparse.ArgumentParser()

#parser.add_argument("number", type=int, help="an integer number")
#input_args = parser.parse_args()
#result = input_args.number * 2
#print("Result:", result)

output_res = []
no_of_iterations_run = 1
for iterations in  range(no_of_iterations_run) :
    # 1. get the digit dataset using our defined function
    X, y = read_digits()
    
    test_size = [0.1, 0.2, 0.3, 0.4]
    dev_size = [0.1, 0.2, 0.3, 0.4]

    # run iteration for each defined test and dev size splits

    for i in test_size :
        for j in dev_size : 

            #split the dataset into train dev and test set

            X_train, X_test,X_dev, y_train, y_test, y_dev = split_train_dev_test(X, y, test_size=i, dev_size=j)
            
            # pre process the splitted data set
            X_train = preprocessing_data(X_train)

            X_dev = preprocessing_data(X_dev)

            X_test = preprocessing_data(X_test)
            
            
            # training the data set on a different possible hyper parameters  

            gamma_list = [0.0001, 0.0005, 0.001, 0.01, 0.1, 1]
            C_list = [0.1, 1, 10, 100, 1000]
            
            # define svm ml model params
            svm_parameters ={}
            svm_parameters = {'gamma': gamma_list,'C' : C_list,'kernel' :['rbf','linear']}
            
            # define the tree ml model algos

            max_depth_list = [5, 10, 15, 20, 50, 100]
            tree_parameters = {}
            tree_parameters['max_depth'] = max_depth_list

            solver_list=['lbfgs', 'liblinear', 'newton-cg', 'saga']
            logistic_parameters={}
            logistic_parameters['solver']=solver_list

            # ml_model_algorithms = ['tree','svm']
            ml_model_algorithms = ['logisticRegression', 'tree','svm']


            
            # for each model algorithm train a model and save the best model 

            for ml_model_type in ml_model_algorithms:

                # for decesion tree model 
                if ml_model_type =="tree" :

                    # tune the hyper parameter and capture the best scored trained model

                    best_hyperparams, best_scored_model_path, best_model_accuracy = tune_hparams_tree( X_train, y_train,X_dev, y_dev,tree_parameters,model_type ='tree')
                    
                    # loading the best trained model        
                    best_trained_model = load(best_scored_model_path)
                    
                    #evaluate the best trained model 
                    train_accuracy,_ = predict_and_eval(best_trained_model, X_train, y_train)
                    dev_accuracy,_ = predict_and_eval(best_trained_model, X_dev, y_dev)
                    test_accuracy,_ = predict_and_eval(best_trained_model, X_test, y_test)
                    
                    current_model_output1 = {'ml_model_type': ml_model_type, 'iterations': iterations, 'train_accuracy' : train_accuracy, 'dev_accuracy': dev_accuracy, 'test_accuracy': test_accuracy}
                    output_res.append(current_model_output1)

                    print("Accuracy_test  : {0:2f}% Accuracy_dev  : {1:2f}% Accuracy_train  : {2:2f}% ".format((test_accuracy*100),(dev_accuracy*100),(train_accuracy*100)))
                
                # now training a support vector machine model 
                if ml_model_type == "svm" :

                     # tune the hyper parameter and capture the best scored trained model
                    best_hyperparams, best_scored_model_path, best_model_accuracy = tune_hparams_svm( X_train, y_train,X_dev, y_dev,svm_parameters,model_type ='svm')
               
                    
                    # load the best trained model         
                    best_trained_model = load(best_scored_model_path)
                    
                    #evaluate the best trained model 
                    train_accuracy,_ = predict_and_eval(best_trained_model, X_train, y_train)
                    dev_accuracy,_ = predict_and_eval(best_trained_model, X_dev, y_dev)
                    test_accuracy,_ = predict_and_eval(best_trained_model, X_test, y_test)
                    
                    current_model_output2 = {'ml_model_type': ml_model_type, 'iterations': iterations, 'train_accuracy' : train_accuracy, 'dev_accuracy': dev_accuracy, 'test_accuracy': test_accuracy}
                    output_res.append(current_model_output2)
                    
                    print("Accuracy_test  : {0:2f}% Accuracy_dev  : {1:2f}% Accuracy_train  : {2:2f}% ".format((test_accuracy*100),(dev_accuracy*100),(train_accuracy*100)))
                # load the best of both model          
                #best_trained_model = load(best_scored_model_path)

                if ml_model_type == "logisticRegression" :

                # tune the hyper parameter and capture the best scored trained model
                    best_hyperparams, best_scored_model_path, best_model_accuracy = tune_hparams_svm( X_train, y_train,X_dev, y_dev,logistic_parameters,model_type ='lr')
                
                    
                    # load the best trained model         
                    best_trained_model = load(best_scored_model_path)
                    
                    #evaluate the best trained model 
                    train_accuracy,_ = predict_and_eval(best_trained_model, X_train, y_train)
                    dev_accuracy,_ = predict_and_eval(best_trained_model, X_dev, y_dev)
                    test_accuracy,_ = predict_and_eval(best_trained_model, X_test, y_test)
                    
                    current_model_output3 = {'ml_model_type': ml_model_type, 'iterations': iterations, 'train_accuracy' : train_accuracy, 'dev_accuracy': dev_accuracy, 'test_accuracy': test_accuracy}
                    output_res.append(current_model_output3)
                    
                    print("Accuracy_test  : {0:2f}% Accuracy_dev  : {1:2f}% Accuracy_train  : {2:2f}% ".format((test_accuracy*100),(dev_accuracy*100),(train_accuracy*100)))
                # load the best of both model          
                #best_trained_model = load(best_scored_model_path)
    
                print('Train : {0} Test_size : {1} Dev_size :{2}'.format( round(1-i-j,1) , i,  j)," ", end = "")
                
                print(pd.DataFrame(output_res).groupby('ml_model_type').describe().T)
