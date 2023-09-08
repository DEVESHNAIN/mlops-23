import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

# read the input data
def read_digits():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    return X, y 

# preprocess the input data 
def preprocess_data(data):
    # flatten the images
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

# Split dataframe into 3 dataframes- train, dev and test
def split_train_dev_test(X, y, test_size, dev_size):
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(X, y, test_size=(test_size + dev_size), shuffle=False)
    test_size_in_dev_test=dev_size / test_size + dev_size
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
def train_model(x, y, model_params, model_type="svm"):
    if model_type == "svm":
        # Create a classifier: a support vector classifier
        clf = svm.SVC
    model = clf(**model_params)
    # train the model
    model.fit(x, y)
    return model

# predict and aval function for hyper parameter tuning

def predict_and_eval(model, X_test, y_test):

    predicted = model.predict(X_test)
    acc=metrics.accuracy_score(y_test, predicted)
    print("Accuracy is ", acc)
    return acc