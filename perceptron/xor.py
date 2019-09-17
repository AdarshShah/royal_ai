import numpy as np 
from perceptron import Perceptron

inputs = []
for i in [0,1]:
    for j in [0,1]:
        inputs.append(np.array([i,j]))


outputs = np.array([1,0,0,0])
h1 = Perceptron(2)
h1.train(inputs,outputs)

outputs = np.array([0,1,0,0])
h2 = Perceptron(2)
h2.train(inputs,outputs)

outputs = np.array([0,0,1,0])
h3 = Perceptron(2)
h3.train(inputs,outputs)

outputs = np.array([0,0,0,1])
h4 = Perceptron(2)
h4.train(inputs,outputs)

inputs = [np.array([1,0,0,0]),np.array([0,1,0,0]),np.array([0,0,1,0]),np.array([0,0,0,1])]
outputs = np.array([0,1,1,0])
h5 = Perceptron(4)
h5.train(inputs,outputs)

x = int(input("Enter x : "))
y = int(input("Enter y : "))
l1 = [h1.predict([x,y]),h2.predict([x,y]),h3.predict([x,y]),h4.predict([x,y])]
print(h5.predict(l1))


