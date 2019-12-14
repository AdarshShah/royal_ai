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

def do_momentum_gradient_descent():
    w, b, eta = init_w,init_b, 1.0
    prev_v_w, prev_v_b, gamma = 0, 0, 0.9
    for i in range(max_epochs):
        dw, db = 0, 0
        for x,y in zip(X,Y):
            dw += gradient_w(w, b , x , y)
            db += gradient_b(w, b , x , y)
        v_w = gamma * prev_v_w + eta * dw
        v_b = gamma * prev_v_b + eta * db
        w = w - v_w
        b = b - v_b
        prev_v_w = v_w                    
        prev_v_b = v_b

def do_nesterov_gradient_descent():
    w, b, eta = init_w, init_b, 1.0
    prev_v_w, prev_v_b, gamma = 0, 0, 0.9
    for i in range(max_epochs):
        dw, db = 0, 0
        v_w = gamma * prev_v_w
        v_b = gamma * prev_v_b
        for x,y in zip(X,Y):
            dw += grad_w(w*v_w,b*v_b,x,y)
            db += grad_b(w*v_b,b*v_b,x,y)
        v_w = gamma * prev_v_w + eta * dw            
        v_b = gamma * prev_v_b + eta * db