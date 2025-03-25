# Program to calculate the apple prizes and tax using the forward and backpropagation using the naive layer
# coding: utf-8
# Import the naive layer module
# by Wilton Machimbira

from layer_naive import *


apple = 100
apple_num = 2
tax = 1.1

# Instantiate the multiplication layer from the naive module 
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# Perform forward Propagation
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

# Perform backward propagation
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

# Print the prices for apples and tax
print("price:", int(price))
print("dApple:", dapple)
print("dApple_num:", int(dapple_num))
print("dTax:", dtax)
