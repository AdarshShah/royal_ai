import numpy as np
import math

X = [0.5,2.5]
Y = [0.2,0.9]

def sigmoid(w,b,x):
    return 1.0 / (1.0 + math.exp(-(w*x+b)))

def error(w,b):
    err = 0.0
    for x,y in zip(X,Y):
        fx = sigmoid(w,b,x)
        err += 0.5 * (fx-y)**2
    return err

def gradient_b(w,b,x,y):
    fx = sigmoid(w,b,x)
    return (fx-y)*fx*(1-fx)

def gradient_w(w,b,x,y):
    fx = sigmoid(w,b,x)
    return (fx-y)*fx*(1-fx)*x

def do_gradient_descent():
    w, b, eta, max_epochs = -2, -2, 1.0, 1000
    for i in range(max_epochs):
        dw, db = 0, 0
        for x,y in zip(X,Y):
            dw += gradient_w(w,b,x,y)
            db += gradient_b(w,b,x,y)
        w = w - eta * dw
        b = b - eta * db