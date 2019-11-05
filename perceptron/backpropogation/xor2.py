import numpy as np
import math


# input = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1]])
# output = np.array(([0],[1],[1],[0],[1],[0]))

input = np.array([[0,0],[0,1],[1,0],[1,1]])
output = np.array([0,1,1,0])
eta = 4

W_1 = np.ones((2,2))
b_1 = np.ones((2,1))
a_1 = np.ones((2,1))
h_1 = np.ones((2,1))

# W_2 = np.ones((3,6))
# b_2 = np.ones((3,1))
# a_2 = np.ones((3,1))
# h_2 = np.ones((3,1))

W_3 = np.ones((1,2))
b_3 = np.ones((1,1))
a_3 = np.ones((1,1))
h_3 = np.ones((1,1))

def sigmoid(a):
    return 1/(1+np.exp(-a))

def sigmoid_grad(a):
    return a*(1-a)

def feedforward(I):
    global W_1,W_2,W_3,a_1,a_2,a_3,b_1,b_2,b_3,h_1,h_2,h_3,eta
    a_1 = np.dot(W_1,I.T) + b_1
    h_1 = sigmoid(a_1)
    # a_2 = np.dot(W_2,h_1) #+ b_2
    # h_2 = sigmoid(a_2)
    a_3 = np.dot(W_3,h_1) + b_3
    h_3 = sigmoid(a_3)
    return h_3

def backpropogation(I,O):
    global W_1,W_2,W_3,a_1,a_2,a_3,b_1,b_2,b_3,h_1,h_2,h_3,eta
    feedforward(I)
    G = h_3 - O 
    G = G*sigmoid_grad(h_3)
    print(G) #1*4

    print(h_1.T)
    G_W_3 = np.dot(G,h_1.T)
    print(G_W_3)
    G_b_3 = np.sum(G,1)
    print(G_b_3)
    print(G)
    print(W_3)
    print(sigmoid_grad(h_1))
    G = np.dot(G.T,W_3).T #4*2
    print(G)


    # G_W_2 = np.dot(G,h_1.T) #2X4
    # G_b_2 = G
    # G = np.dot(W_2.T,G)*sigmoid_grad(h_1) #4X1

    #print(I)
    G_W_1 = np.dot(G,I) 
    G_b_1 = np.array([np.sum(G,1)])
    #print(G_b_1)

    W_1 -= eta*G_W_1
    b_1 -= eta*G_b_1.T

    # W_2 += eta*G_W_2
    # b_2 += eta*G_b_2

    W_3 -= eta*G_W_3
    b_3 -= eta*G_b_3

def lossfunction():
    global input,output
    sum = 0
    for i,o in zip(input,output):
        sum += (feedforward(np.array([i]))-o)**2
    return sum


backpropogation(input,output)
  


# for _ in range(1000):
#     backpropogation(input,output)
#     if _ % 100 == 0:
#         print(lossfunction())
    


# for i,o in zip(input,output):
#     print(o)
#     print(feedforward(np.array([i])))


    
    

