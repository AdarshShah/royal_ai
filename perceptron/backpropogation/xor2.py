import numpy as np 
import math

input = np.array([[0,0],[0,1],[1,0],[1,1]])
output = np.array([0,1,1,0])
eta = 1


W_1 = np.ones((4,2))
b_1 = np.ones((4,1))
a_1 = np.ones((4,1))
h_1 = np.ones((4,1))

W_2 = np.ones((4,4))
b_2 = np.ones((4,1))
a_2 = np.ones((4,1))
h_2 = np.ones((4,1))

W_3 = np.ones((1,4))
b_3 = np.ones((1,1))
a_3 = np.ones((1,1))
h_3 = np.ones((1,1))

def sigmoid(a):
    return 1/(1+np.exp(-1*a))

def sigmoid_grad(a):
    x = sigmoid(a)
    x = x*(1-x)
    return x

def feedforward(I):
    global W_1,W_2,W_3,a_1,a_2,a_3,b_1,b_2,b_3,h_1,h_2,h_3,eta
    a_1 = W_1.dot(I.T) + b_1
    h_1 = sigmoid(a_1)
    a_2 = W_2.dot(h_1) + b_2
    h_2 = sigmoid(a_2)
    a_3 = W_3.dot(h_2) + b_3
    h_3 = sigmoid(a_3)
    return h_3

def backpropogation(I,O):
    global W_1,W_2,W_3,a_1,a_2,a_3,b_1,b_2,b_3,h_1,h_2,h_3,eta
    G = feedforward(I) - O 
    G = G*sigmoid_grad(a_3)

    G_W_3 = G*h_2.T
    G_b_3 = G
    G = W_3.T.dot(G)*sigmoid_grad(a_2) #2X1

    G_W_2 = G.dot(h_1.T) #2X4
    G_b_2 = G
    G = W_2.T.dot(G)*sigmoid_grad(a_1) #4X1

    G_W_1 = G.dot(I) 
    G_b_1 = G

    W_1 = W_1 - eta*G_W_1
    b_1 = b_1 - eta*G_b_1

    W_2 = W_2 - eta*G_W_2
    b_2 = b_2 - eta*G_b_2
    
    W_3 = W_3 - eta*G_W_3
    b_3 = b_3 - eta*G_b_3


for _ in range(7000):
    for i,o in zip(input,output):
        feedforward(np.array([i]))
        backpropogation(np.array([i]),o)


for i,o in zip(input,output):
    print(feedforward(np.array([i])))
    backpropogation(np.array([i]),o)
    

