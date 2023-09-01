import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split


# Split dataframe into 3 dataframes- train, dev and test
def split_train_dev_test(X, y, test_size, dev_size):
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(X, y, test_size=(test_size + dev_size), shuffle=False)
    test_size_in_dev_test=dev_size / test_size + dev_size
    X_dev, X_test, y_dev, y_test = train_test_split(X_dev_test, y_dev_test, test_size=test_size_in_dev_test,
                                                    shuffle=False)
    return X_train, X_dev, X_test, y_train, y_dev, y_test


#this function is to predict and evaluate the model performance
def predict_and_eval(model, X_test, y_test):
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