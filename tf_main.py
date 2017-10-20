import numpy as np
from tf_utils import load_dataset, preprocess_data
from tf_app_utils import model, predict
import pickle

np.random.seed(1)

# Parameters
num_classes = 6

X_train_orig_0, Y_train_orig_0, X_test_orig_0, Y_test_orig_0, classes = load_dataset(
'train_signs.h5',
'test_signs.h5')

X_train_orig_1, Y_train_orig_1, X_test_orig_1, Y_test_orig_1, _ = load_dataset(
'train_signs_1.h5',
'test_signs_1.h5')

X_train_orig_2, Y_train_orig_2, X_test_orig_2, Y_test_orig_2, _ = load_dataset(
'train_signs_2.h5',
'test_signs_2.h5')

X_train_orig_3, Y_train_orig_3, X_test_orig_3, Y_test_orig_3, _ = load_dataset(
'train_signs_3.h5',
'test_signs_3.h5')

X_train_orig = np.append(X_train_orig_0, X_train_orig_1, axis = 0)
X_train_orig = np.append(X_train_orig, X_train_orig_2, axis = 0)
X_train_orig = np.append(X_train_orig, X_train_orig_3, axis = 0)

Y_train_orig = np.append(Y_train_orig_0, Y_train_orig_1, axis = 1)
Y_train_orig = np.append(Y_train_orig, Y_train_orig_2, axis = 1)
Y_train_orig = np.append(Y_train_orig, Y_train_orig_3, axis = 1)

X_test_orig = np.append(X_test_orig_0, X_test_orig_1, axis = 0)
X_test_orig = np.append(X_test_orig, X_test_orig_2, axis = 0)
X_test_orig = np.append(X_test_orig, X_test_orig_3, axis = 0)

Y_test_orig = np.append(Y_test_orig_0, Y_test_orig_1, axis = 1)
Y_test_orig = np.append(Y_test_orig, Y_test_orig_2, axis = 1)
Y_test_orig = np.append(Y_test_orig, Y_test_orig_3, axis = 1)

# Cleaning the dataset
X_train, Y_train, X_test, Y_test = preprocess_data(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, num_classes)

# Training the parameters
parameters = model(X_train, Y_train, X_test, Y_test)

# Store parameters
output = open('best_parameters.pkl','wb')
pickle.dump(parameters, output)

#Predict using a sample image of test dataset
#predicted_class = predict(X_test[:,1:2], parameters, X_test.shape[0])