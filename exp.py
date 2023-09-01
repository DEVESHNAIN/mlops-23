"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from utils import split_train_dev_test,predict_and_eval

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.


# 1. Get the dataset
digits = datasets.load_digits()

# 2. Qualitative sanity check of the data
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.


# 3. Data preprocessing
# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# 4. Data splitting -- to create train, dev and test sets
# Split data into 50% train , 20% dev and 30% text subsets

X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(
    data, digits.target, test_size=0.3,dev_size=0.2
)

# 5. Model training
# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)
# Learn the digits on the train subset
clf.fit(X_train, y_train)

# # 6. Getting model predictions on dev set
# # Predict the value of the digit on the dev subset

# predict_and_eval(clf,X_dev,y_dev)

# # 6. Getting model predictions on test set
# # Predict the value of the digit on the test subset

predict_and_eval(clf,X_test,y_test)
