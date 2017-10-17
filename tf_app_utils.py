import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import *

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_x -- scalar, size of an image vector (height * width * depth)
    n_y -- scalar, number of classes

    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

    """

    X = tf.placeholder(tf.float32, shape=[n_x, None])
    Y = tf.placeholder(tf.float32, shape=[n_y, None])

    return X, Y


def initialize_parameters(layer_dims):
    """
    Initializes parameters to build a neural network with tensorflow.
    
    Returns:
    parameters -- a dictionary of tensors containing weights and biases
    """

    tf.set_random_seed(1)
    L = len(layer_dims) - 1
    parameters = {}

    for l in range(L):
        parameters["W"+str(l+1)] = tf.get_variable("W"+str(l+1),
                                                   [layer_dims[l+1], layer_dims[l]],
                                                   initializer=tf.contrib.layers.xavier_initializer(seed = 1))

        parameters["b"+str(l+1)] = tf.get_variable("b"+str(l+1),
                                                   [layer_dims[l+1], 1],
                                                   initializer=tf.zeros_initializer())

    return parameters


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: (L-1)[LINEAR->RELU]->LINEAR->SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing model parameters

    Returns:
    ZL -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    L = len(parameters) // 2
    A = X
    Z = None

    for l in range(L):
        W = parameters["W"+str(l+1)]
        b = parameters["b"+str(l+1)]

        Z = tf.add(tf.matmul(W, A), b)
        A = tf.nn.relu(Z)

    return Z


def compute_cost(ZL, Y):
    """
    Computes the cost

    Arguments:
    ZL -- output of forward propagation (output of the last LINEAR unit), of shape (number of classes, number of examples)
    Y -- "true" labels vector placeholder, same shape as ZL

    Returns:
    cost - Tensor of the cost function
    """

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(ZL)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=labels))

    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          num_epochs=1500, minibatch_size=32, print_cost=True):
    """
    Implements a L-layer tensorflow neural network: (L-1)[LINEAR->RELU]->LINEAR->SOFTMAX

    Arguments:
    X_train -- training set, of shape (input size, number of training examples)
    Y_train -- test set, of shape (output size, number of training examples)
    X_test -- training set, of shape (input size, number of training examples)
    Y_test -- test set, of shape (output size, number of test examples)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep consistent results
    seed = 3  # to keep consistent results
    (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]  # n_y : output size
    costs = []  # To keep track of the cost
    layer_dims = [n_x, 25, 12, n_y] # to initialize parameters

    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters(layer_dims)

    # Forward propagation: Build the forward propagation in the tensorflow graph
    ZL = forward_propagation(X, parameters=parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(ZL, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.  # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters


def predict(X, parameters, n_x):

    L = len(parameters) // 2
    params = {}

    for l in range(L):
        params["W"+str(l+1)] = tf.convert_to_tensor(parameters["W"+str(l+1)])
        params["b"+str(l+1)] = tf.convert_to_tensor(parameters["b"+str(l+1)])

    x = tf.placeholder("float", [n_x, 1])

    zl = forward_propagation_for_predict(x, params)
    p = tf.argmax(zl)

    sess = tf.Session()
    prediction = sess.run(p, feed_dict={x: X})

    return prediction


def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: (L-1)[LINEAR -> RELU] -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing parameters

    Returns:
    ZL -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    L = len(parameters) // 2
    A = X
    Z = None

    for l in range(L):
        W = parameters["W"+str(l+1)]
        b = parameters["b"+str(l+1)]

        Z = tf.add(tf.matmul(W, A), b)
        A = tf.nn.relu(Z)

    return Z