# Program that calculate the numerical difference of a curve and produce a graph using matplotlib.
# Name : Wilton Machimbira

# Import the numpy and matplotlib libraries
import numpy as np
import matplotlib.pylab as plt


# Funtion to calculate numerical difference
# h is added

def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
    return 0.01*x**2 + 0.1*x   # f(x) = 0.01 * x^2 +0.1*x

# function to calculate the tangent of the line using numerical difference 
def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y
     
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)

# Plotting the graph of the function and the tangent line.
plt.plot(x, y)
plt.plot(x, y2)
plt.title("Graph of the function f(x) and its Tangent line")
plt.show()
