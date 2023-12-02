import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split,GridSearchCV
import itertools
import numpy as np
import os,pdb
from joblib import dump, load
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import linear_model

# read the input data
def read_digits():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    return X, y 

# preprocess the input data 
def preprocessing_data(data):
    # flatten the images
    n_samples = len(data)
    data = data.reshape((n_samples, -1))

    # normalize the samples
    normalized_data=preprocessing.normalize(data)

    return normalized_data

# Split dataframe into 3 dataframes- train, dev and test
def split_train_dev_test(X, y, test_size, dev_size):
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(X, y, test_size=(test_size + dev_size), shuffle=False)
    test_size_in_dev_test=dev_size / (test_size + dev_size)
    X_dev, X_test, y_dev, y_test = train_test_split(X_dev_test, y_dev_test, test_size=test_size_in_dev_test,
                                                    shuffle=False)
    return X_train, X_dev, X_test, y_train, y_dev, y_test


#this function is to predict and evaluate the model performance
def predict_and_eval_old(model, X_test, y_test):
    predicted = model.predict(X_test)
    print(f"Classification report for classifier  {model}  model :\n{metrics.classification_report(y_test, predicted)}\n")

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    y_true = []
    y_pred = []
    cm = disp.confusion_matrix

    for gt in range(len(cm)):
        for pred in range(len(cm)):
            y_true += [gt] * cm[gt][pred]
            y_pred += [pred] * cm[gt][pred]

    print("Classification report rebuilt from confusion matrix:\n"
          f"{metrics.classification_report(y_true, y_pred)}\n")
    plt.show()


# function to train the model 
# def train_model(x, y, model_params, model_type="svm"):
#     if model_type == "svm":
#         # Create a classifier: a support vector classifier
#         clf = svm.SVC
#     model = clf(**model_params)
#     # train the model
#     model.fit(x, y)
#     return model

# predict and aval function for hyper parameter tuning

def predict_and_eval(model, X_test, y_test):

    predicted = model.predict(X_test)
    acc=metrics.accuracy_score(y_test, predicted)
    # print("Accuracy is ", acc)
    f1_value = metrics.f1_score(y_test, predicted, average="macro")
    return acc, f1_value



def tune_hparams(X_train, y_train, X_dev, y_dev, list_of_all_param_comb):

    # initialize the metrics
    best_acc_so_far = -1
    best_model = None
    best_hyperparams = None
    best_train_acc = -1

    #for each comb train a model and capture dev accuracy
    for param_comb in list_of_all_param_comb:

        # model_params = {"gamma": param_comb[0], "C": param_comb[1]}
        model = train_model(X_train, y_train, param_comb)
        cur_dev_accuracy = predict_and_eval(model, X_dev, y_dev)

        # If dev accuracy is more than best captured accuracy, update the best hyper parameters
        if cur_dev_accuracy > best_acc_so_far:
            best_acc_so_far = cur_dev_accuracy
            best_model = model
            best_hyperparams = param_comb

    return best_hyperparams, best_model, best_acc_so_far



def get_param_combs(name_of_param, value_of_param, base_param_comb):    
    new_comb = []

    for value in value_of_param:

        for comb in base_param_comb:

            comb[name_of_param] = value
            new_comb.append(comb.copy()) 

    return new_comb


def fetch_hyperparameter_combinations(param_list_dict):    
    base_comb = [{}]

    for name_of_param, value_of_param in param_list_dict.items():

        base_comb = get_param_combs(name_of_param, value_of_param, base_comb)

    return base_comb

def tune_hparams_svm( X_train, Y_train, x_dev, y_dev,list_of_all_param_combination,model_type="svm"):
    
    # create all combinations of hyper parameters
    keys, values = zip(*list_of_all_param_combination.items())

    combinations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    
    
    # default base setting     
    g = 'scale'
    cval = 1.0
    kval = 'rbf'
    best_model_score =[0,0]
    best_model = None
    avg_scores ={}
    best_hyperparams = {}

    # for each model params train model and check the model score
    for model_params in combinations_dicts :
        
        '''for k,v in dict1.items():
                    
                    if k == 'gamma':
                        g = v
                    elif k =='C':
                        cval = v
                    elif k == 'kernels':
                        kval = v'''

        # trained model for selected hyper parameters
        current_model = train_model(X_train, Y_train, model_params, model_type=model_type)
        
        # evaluate trained model
        current_model_scores, _  = predict_and_eval(current_model, x_dev, y_dev)
        
        # calculating average of model scores
        avg_scores[str(kval) + "-" + str(cval)+ "-" + str(g)]= round(np.average(current_model_scores),3)
        best_model_path = ""

        # compare the current model with best model, if better, replace the best model

        if best_model_score[1] < round(np.average(current_model_scores),3) :
            best_model = current_model
            best_model_score[1] = round(np.average(current_model_scores),3)
            best_model_score[0] = str(kval) + "-" + str(cval)+ "-" + str(g)
            best_hyperparams = model_params
            y_pred_train =best_model.predict(X_train)
            best_accuracy = best_model_score[1]

    
    os.makedirs('models', exist_ok=True)
    # define model name based on hyper parameters and model path
       
    model_path = "./models/M22AIE247_{}_".format(model_type) +"_".join(["{}_{}".format(k,v) for k,v in best_hyperparams.items()]) + ".joblib"
    print(model_path)
    # dump the best_model    
    dump(best_model, model_path)

    return best_hyperparams, model_path, best_accuracy

# model training  function for specific parameters
def train_model(x, y, model_params, model_type="svm"):

    # training svm or tree based model
    if model_type == "svm":
        clf = svm.SVC
        model = clf(**model_params)

        model.fit(x, y)

    if model_type == "tree":
        clf = tree.DecisionTreeClassifier
        
        model = clf(**model_params)

        model.fit(x, y)
    
    if model_type == "lr":
        clf = linear_model.LogisticRegression
        
        model = clf(**model_params)

        model.fit(x, y)
    return model    


# function to tune hyper parameters for the tree based model similar to the svm model
def tune_hparams_tree( X_train, Y_train,x_dev, y_dev,list_of_all_param_combination,model_type ='tree'):
    
    # generating the all combinations of hyper parameters
    keys, values = zip(*list_of_all_param_combination.items())

    combinations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    
    # default base configuration     
    best_hyperparams = {}
    best_model_score =[0,0]
    best_model = None
    avg_scores ={}
    for model_params in combinations_dicts :
        
        # train model for each hyper parameters
        current_model = train_model(X_train, Y_train, model_params, model_type="tree")
        # predict the score for trained model 
        current_model_scores = predict_and_eval(current_model, x_dev, y_dev)
       
       # compare the model performance with the base model 
        if best_model_score[1] < round(np.average(current_model_scores),3) :
            #intially best model is current model 
            best_model = current_model
            # averaging the current model scores
            best_model_score[1] = round(np.average(current_model_scores),3)
            #best_model_score[0] = str(kval) + "-" + str(cval)+ "-" + str(g)
            best_hyperparams = model_params
            best_accuracy = best_model_score[1]
    
    # defining the model path and model name wrt to the hyper parameters of the model 
    model_path = "./models/M22AIE247_{}_".format(model_type) +"_".join(["{}_{}".format(k,v) for k,v in best_hyperparams.items()]) + ".joblib"
    print(model_path)
    # save the best_model    
    dump(best_model, model_path)
   
    return best_hyperparams, model_path, best_accuracy


def tune_hparams_logistic( X_train, Y_train,x_dev, y_dev,list_of_all_param_combination,model_type ='lr'):
    
    # generating the all combinations of hyper parameters
    keys, values = zip(*list_of_all_param_combination.items())

    combinations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    
    # default base configuration     
    best_hyperparams = {}
    best_model_score =[0,0]
    best_model = None
    avg_scores ={}
    for model_params in combinations_dicts :
        
        # train model for each hyper parameters
        current_model = train_model(X_train, Y_train, model_params, model_type="lr")
        # predict the score for trained model 
        current_model_scores = predict_and_eval(current_model, x_dev, y_dev)
       
       # compare the model performance with the base model 
        if best_model_score[1] < round(np.average(current_model_scores),3) :
            #intially best model is current model 
            best_model = current_model
            # averaging the current model scores
            best_model_score[1] = round(np.average(current_model_scores),3)
            #best_model_score[0] = str(kval) + "-" + str(cval)+ "-" + str(g)
            best_hyperparams = model_params
            best_accuracy = best_model_score[1]
    
    # defining the model path and model name wrt to the hyper parameters of the model 
    model_path = "./models/M22AIE247_{}_".format(model_type) +"_".join(["{}_{}".format(k,v) for k,v in best_hyperparams.items()]) + ".joblib"
    print(model_path)
    # save the best_model    
    dump(best_model, model_path)
   
    return best_hyperparams, model_path, best_accuracy