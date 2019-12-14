import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, Activation
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import mnist

train_images = mnist.download_and_parse_mnist_file(fname="train-images-idx3-ubyte.gz")
train_images = np.reshape(train_images,(len(train_images),784))
standardScaler = StandardScaler()
train_images = standardScaler.fit_transform(train_images)

test_images = mnist.download_and_parse_mnist_file(fname="t10k-images-idx3-ubyte.gz")
test_images = np.reshape(test_images,(len(test_images),784))
test_images = standardScaler.transform(test_images)

train_labels = mnist.download_and_parse_mnist_file(fname="train-labels-idx1-ubyte.gz")
train_labels = np.reshape(train_labels,(-1,1))

test_labels = mnist.download_and_parse_mnist_file(fname="t10k-labels-idx1-ubyte.gz")

ohe = OneHotEncoder()
train_labels = ohe.fit_transform(train_labels).toarray()

model = Sequential()
model.add(Dense(units=32,input_shape=(784,),activation="relu")) 
model.add(Dense(units=10,activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer="adam")
model.fit(train_images,train_labels,epochs=10,batch_size=32)

predict = model.predict(test_images)

predict = np.dot(predict,ohe.active_features_)

print(test_labels)
print(predict)


