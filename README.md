An algorithm that facilitates communication between a speech-impaired person and someone who doesn't understand sign language.

Training set: 1080 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (180 pictures per number).

Test set: 120 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (20 pictures per number).

Here are examples for each number, and corresponding labels converted to one-hot. 
![alt signs_dataset](https://raw.githubusercontent.com/tejaslodaya/tensorflow-signs-nn/master/signs_dataset.png)

Architecture:
1. Input is an image of size 64x64x3 (RGB), which is flattened to shape 12288 and normalized it by dividing it by 255
2. Hidden layers of size (12288 -> 25 -> 12 -> 6)
3. The output of last hidden layer gives a probability of the image belonging to one of the six classes
4. RELU activation function. Cross entropy cost. Adam optimizer
5. Mini-batch gradient descent with minibatch_size of 32

The model is LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX.

Outcome:

1.  Training cost graph-

![alt cost](https://raw.githubusercontent.com/tejaslodaya/tensorflow-signs-nn/master/cost.png)
2.  Train Accuracy - 0.999074 <br>
    Test Accuracy - 0.716667
3.  TODO- to overcome overfitting, add L2 or dropout regularization    