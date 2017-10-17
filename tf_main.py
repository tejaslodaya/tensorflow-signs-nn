import numpy as np
from tf_utils import load_dataset, preprocess_data
from tf_app_utils import model, predict

np.random.seed(1)

# Parameters
num_classes = 6

# Loading the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Cleaning the dataset
X_train, Y_train, X_test, Y_test = preprocess_data(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, num_classes)

# Training the parameters
parameters = model(X_train, Y_train, X_test, Y_test, num_epochs=100)

#Predict using a sample image of test dataset
predicted_class = predict(X_test[:,1:2], parameters, X_test.shape[0])