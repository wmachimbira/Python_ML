# coding: utf-8
# Import matplotlib.pyplot and numpy
# import and load mnist dataset
# Import and load TwoLayerNet 
# Name  : Wilton Machimbira
import sys, os
sys.path.append(os.pardir)  # Import/set to import files from the directory
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# Create a TwoLayerNetwork with input size 784, hidden size  50 and output size 10

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
# Train the model for 10000 iterations
# 하이퍼파라미터
iters_num = 10000  # set the number of iterations to 10 000 .
train_size = x_train.shape[0]
batch_size = 100   # Mini batch size
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# Calculating the epoch (number of iterations per ephemeral)
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # Mini batch acquisition
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # Calculate the slope
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # Update the parameters
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # Calculating accuracy
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # Calculate accuracy per ephemeral
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# Train the Neural Network
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.title("Graph of the accuracy vs Epoch of a Two layer Neural Network")
plt.show()
