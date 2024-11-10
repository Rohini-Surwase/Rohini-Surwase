build the image classification model by dividing the model into following 4 stages: 
a. Loding and preprocessing data 
b. defining the models architecture 
c. training the model 
d. Estimating the models performance

# DL ASSIGNMENT NO 3

# Importing Required Packages

import tenserflow as tf
from keras.model import sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

#Loading and preprossing the image data

mist = tf.keras.datasets.mist
(x_train,y_train),(x_test,y_test)=mist.load_data()
input_shape = (28,28,1)

#making sure that the value are float so that we can get the decimal points after division

print('Data type of x_train: ', x_test.dtype)
x_train = x_train.reshape([0],28,28,1)
x_test = x_test.reshape([0],28,28,1)

print('Data type after converting to the float ', x_train.dtype)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalizing the RGB codes by dividing it to the max RGB value

x_train = x_train / 255
x_test = x_test / 255
print('Shape of Training :', x_train.shape)
print('Shape of Testing :', x_train.shap)

# DEFINING THE MODEL'S ARCHITECTURE 

model = sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pooling_size=(2,2)))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

model.summary()


# TRAINING THE MODEL

model.compile(optimizer='adam', loss='sparse_categorial_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=2)

# ESSTIMATING THE MODELS PERFORMANCE 

test_loss, test_acc = model.evaluate(x_test,y_test)
print('loss = %.3f' %test_loss)
print('Accuracy = %.3f' %test_acc)

#showing image at the position [] from dataset

image= x_train[0]
plt.imshow(np.squeeze(image), cmap='gray')
plt.show()

# predicting the class of the image

image= image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
predict_model= model.predict([image])
print('predicting class: {}'.format(np.argmax(predict_model)))
