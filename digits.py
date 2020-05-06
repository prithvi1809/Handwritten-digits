#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os 
import pandas as pd
import seaborn as sns
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical



# In[2]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[3]:


print(x_train[1].shape)


# In[4]:


g=sns.countplot(y_train)


# In[5]:


y_train=to_categorical(y_train,num_classes=10)


# In[6]:


x_train=x_train /255.0
x_test=x_test/255.0


# In[7]:


xtrain,x_val,ytrain,y_val=train_test_split(x_train,y_train,test_size=0.2,random_state=2)


# In[8]:


xtrain.shape


# In[9]:


g=plt.imshow(xtrain[10,:,:])


# In[10]:


x_train[1];
xtrain=xtrain.reshape(48000,28,28,1)
xtrain.shape


# In[11]:


model=Sequential()
model.add(Conv2D(activation='relu',filters=32,kernel_size=(5,5),padding='Same',input_shape=(28,28,1)))
model.add(Conv2D(activation='relu',filters=32,kernel_size=(5,5),padding='Same'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model=Sequential()
model.add(Conv2D(activation='relu',filters=64,kernel_size=(3,3),padding='Same'))
model.add(Conv2D(activation='relu',filters=64,kernel_size=(3,3),padding='Same'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))


# In[12]:


optimizer=Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])


# In[13]:


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(xtrain)


# In[14]:


model.fit(xtrain,ytrain,epochs=1)


# In[16]:


x_val=x_val.reshape(12000,28,28,1)


# In[17]:


history=model.fit(datagen.flow(xtrain,ytrain),epochs=3,validation_data=(x_val,y_val))


# In[19]:


import csv
final_test=pd.read_csv('test.csv')
final_test.head()


# In[20]:


data3 = np.genfromtxt('test.csv', skip_header=1)


# In[21]:


final_test.shape


# In[22]:


final_test=final_test/255.0
final_test=final_test.values.reshape(-1,28,28,1)


# In[23]:


final_test.shape


# In[33]:


score1=model.predict_proba(final_test)


# In[34]:


score1.shape


# In[26]:


score1


# In[35]:


score1


# In[36]:


score1[100]


# In[37]:


score1 = np.argmax(score1,axis=1)
score1 = pd.Series(score1,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),score1],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv",index=False)


# In[ ]:




