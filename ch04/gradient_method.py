# coding: utf-8
# Program to calculate the gradient descent of a curve
#Name : Wilton Machimbira

# import matplotlib and paylab

import numpy as np
import matplotlib.pylab as plt
from gradient_2d import numerical_gradient

# Function to calculate the gradient descent  using 100 steps
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)


def function_2(x):
    return x[0]**2 + x[1]**2 # f(x) =(x0)^2 + (x1)^2

init_x = np.array([-3.0, 4.0])    

lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
("Graph showing gradient descent")
plt.show()
