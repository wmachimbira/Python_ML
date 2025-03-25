# Python Program for naive implementation with multiplication and addition
# coding: utf-8
# Wilton Machimbira

# Multiplication layer
class MulLayer:
    def __init__(self):
        self.x = None # Initialise variables
        self.y = None
# forward pass
    def forward(self, x, y):
        self.x = x
        self.y = y                
        out = x * y # output of multiplication

        return out
# backward pass(Calculating gradients with respect of x,y)
    def backward(self, dout):
        dx = dout * self.y  # x와 y를 바꾼다.
        dy = dout * self.x

        return dx, dy

# addition layer
class AddLayer:
    def __init__(self):
        pass
# Forward pass calculates the sum of two inputs
    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy
