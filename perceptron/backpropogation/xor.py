import numpy as np 

input = np.array([[0,0],[0,1],[1,0],[1,1]])
output = np.array([[0],[1],[1],[0]])

class NeuralNetwork(object):

    def __init__(self):
        self.W_1 = np.random.rand(2,2)
        self.b_1 = np.random.rand(1,2)

        self.W_2 = np.random.rand(2,1)
        self.b_2 = np.random.rand(1,1)

    def sigmoid(self,a):
        return 1/(1+np.exp(-a))

    def feedforward(self,input):
        self.a_1 = np.dot(input,self.W_1)+self.b_1
        self.h_1 = self.sigmoid(self.a_1)

        self.a_2 = np.dot(input,self.W_2)+self.b_2
        self.h_2 = self.sigmoid(self.a_2)

        return self.h_2

    def backprop(self,input,output):
        self.feedforward(input)
        d_error_h_2 = self.h_2 - output
        d_error_a_2 = d_error_h_2 * (1-self.h_2) * self.h_2 

        d_error_h_1 = self.W_2.T*d_error_a_2
        d_error_a_1 = d_error_h_1 * self.h_1 * (1-self.h_1)
        
        self.d_error_W_2 = self.h_1.T.dot(d_error_a_2)
        self.d_error_b_2 = d_error_a_2
        
        self.d_error_W_1 = input.T.dot(d_error_a_1)
        self.d_error_b_1 = d_error_a_1

    def train(self,iter,eta,input,output):
        for _ in range(iter):
            for i,o in zip(input,output):
                self.feedforward(i)
                self.backprop(np.array([i]),o)
                self.W_1-=eta*self.d_error_W_1
                self.b_1-=eta*self.d_error_b_1
                self.W_2-=eta*self.d_error_W_2
                self.b_2-=eta*self.d_error_b_2
            if _%100 == 0:
                print(np.sum((self.h_2-output)*(self.h_2-output)))

n = NeuralNetwork()
n.train(1000,2,input,output)
for i,o in zip(input,output):
    print(n.feedforward(i))


