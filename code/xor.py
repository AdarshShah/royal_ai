import numpy as np 

input = np.array([[0,0],[0,1],[1,0],[1,1]])
output = np.array([[0],[1],[1],[0]])

class NeuralNetwork(object):

    def __init__(self,input,hidden,output):
        self.input = input
        self.hidden = hidden
        self.output = output

        self.W1 = np.random.rand(self.input,self.hidden)
        self.b1 = np.random.rand(1,self.hidden)
        self.W2 = np.random.rand(self.hidden,self.output)
        self.b2 = np.random.rand(1,self.output)
    
    def sigmoid(self,a):
        return 1/(1+np.exp(-a))

    def sigmoid_deriv(self,a):
        return a*(1-a)

    def feedforward(self,input):
        self.a1 = np.dot(input,self.W1) + self.b1
        self.h1 = self.sigmoid(self.a1)
        self.a2 = np.dot(self.h1,self.W2) + self.b2
        self.h2 = self.sigmoid(self.a2)
        return self.h2

    def backprop(self,input,output,eta):
        self.feedforward(input)
        error = self.h2 - output
        d_error_a2 = error*self.sigmoid_deriv(self.h2)

        d_error_W2 = self.h1.T.dot(d_error_a2)
        d_error_b2 = np.sum(d_error_a2,axis=0)
        d_error_a1 = d_error_a2 * self.W2.T * self.sigmoid_deriv(self.h1)

        d_error_W1 = input.T.dot(d_error_a1)
        d_error_b1 = np.sum(d_error_a1,axis=0)

        self.W1 -= eta*d_error_W1
        self.b1 -= eta*d_error_b1
        self.W2 -= eta*d_error_W2
        self.b2 -= eta*d_error_b2

    def lossfunc(self,input,output):
        return np.sum((self.feedforward(input)-output)**2)

        

input = np.array([[0,0],[0,1],[1,0],[1,1]])
output = np.array([[0],[1],[1],[0]])        
n = NeuralNetwork(2,2,1)
for i in range(1000):
    n.backprop(input,output,1)
    if i%50 == 0:
        print(n.lossfunc(input,output))
print(np.round(n.feedforward(input)))

input = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
output = np.array([[0],[1],[1],[0],[1],[0],[0],[1]])
n = NeuralNetwork(3,4,1)
for i in range(2000):
    n.backprop(input,output,1)
    if i%100 == 0:
        print(n.lossfunc(input,output))
print(np.round(n.feedforward(input)))
