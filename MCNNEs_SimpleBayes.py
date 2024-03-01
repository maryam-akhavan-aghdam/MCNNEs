# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 12:38:58 2016

@author: Maryam Akhavan Aghdam

Combibed Mixture of CNN Experts/ Gating = 3/ --> each Gating connected to 3 CNN : nb_output_g = 3
Baysian Method
8-10-12/gate=8
"""

from __future__ import print_function
from __future__ import division
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Merge
from keras.layers import Convolution2D, MaxPooling2D, Input
from keras.utils import np_utils
from keras import backend as K
import cPickle as pickle
import gzip
import numpy as np

from sklearn.metrics import confusion_matrix
from keras.regularizers import l2

batch_size = 22
nb_classes = 2
nb_output_g = 3
nb_epoch = 50
nb_epoch2 = 50

#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
optimizer = 'adamax'
loss = 'binary_crossentropy'
############################# load datasets ################################### 
def load_dataset1():

    filename = 'fMRIX_ABIDEI_Un10.pkl.gz'
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f)
    X_train1, y_train = data[0]
    X_test1, y_test = data[1]
    
    X_train1 = np.array(X_train1)
    print (X_train1.shape)
    y_train = np.array(y_train)
    print (y_train.shape)
    y_train = y_train.reshape((-1,1))
    print (y_train.shape)
    
    X_test1 = np.array(X_test1)
    print (X_test1.shape)
    y_test = np.array(y_test)
    print (y_test.shape)
    y_test = y_test.reshape((-1,1))
    print (y_test.shape)
    
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)
    
    data1 = np.vstack((X_train1,X_test1))
    label1 = np.vstack((y_train,y_test))
    print (data1.shape)
    print (label1.shape)
    
    return data1, label1
    
def load_dataset2():

    filename = 'fMRIY_ABIDEI_Un10.pkl.gz'
    with gzip.open(filename,'rb') as f:
        data = pickle.load(f)
    X_train2, y_train = data[0]
    X_test2, y_test = data[1]

    X_train2 = np.array(X_train2)
    print (X_train2.shape)
    y_train = np.array(y_train)
    print (y_train.shape)
    y_train = y_train.reshape((-1,1))
    print (y_train.shape)
    
    X_test2 = np.array(X_test2)
    print (X_test2.shape)
    y_test = np.array(y_test)
    print (y_test.shape)
    y_test = y_test.reshape((-1,1))
    print (y_test.shape)
    
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)
    
    data2 = np.vstack((X_train2,X_test2))
    label2 = np.vstack((y_train,y_test))
    print (data2.shape)
    print (label2.shape)
    
    return data2, label2
    
    
def load_dataset3():

    filename = 'fMRIZ_ABIDEI_Un10.pkl.gz'
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f)
    X_train3, y_train = data[0]
    X_test3, y_test = data[1]

    X_train3 = np.array(X_train3)
    print (X_train3.shape)
    y_train = np.array(y_train)
    print (y_train.shape)
    y_train = y_train.reshape((-1,1))
    print (y_train.shape)
    
    X_test3 = np.array(X_test3)
    print (X_test3.shape)
    y_test = np.array(y_test)
    print (y_test.shape)
    y_test = y_test.reshape((-1,1))
    print (y_test.shape)
    
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)
    
    data3 = np.vstack((X_train3,X_test3))
    label3 = np.vstack((y_train,y_test))
    print (data3.shape)
    print (label3.shape)
    
    return data3, label3
    
############################################################################### 
# input image dimensions
img_X, img_Y, img_Z = 79, 95, 68
# number of convolutional filters to use
nb_filters1 = 8
nb_filters2 = 10
nb_filters3 = 12

nb_filters_g = 8

# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)


n_folds = 10

data1, label1 = load_dataset1()
data2, label2 = load_dataset2()
data3, label3 = load_dataset3()


part_size1 = int(data1.shape[0]/n_folds)
print (part_size1)
part_size2 = int(data2.shape[0]/n_folds)
print (part_size2)
part_size3 = int(data3.shape[0]/n_folds)
print (part_size3)
################################### softmax ###################################    
## acc
accMOE_fMRIX_soft = np.zeros(shape=(1,n_folds))
accMOE_fMRIX_soft = accMOE_fMRIX_soft.T
print ('accMOE_fMRIX_soft.shape:', accMOE_fMRIX_soft.shape)
accMOE_fMRIY_soft = np.zeros(shape=(1,n_folds))
accMOE_fMRIY_soft = accMOE_fMRIY_soft.T
print ('accMOE_fMRIY_soft.shape:', accMOE_fMRIY_soft.shape)
accMOE_fMRIZ_soft = np.zeros(shape=(1,n_folds))
accMOE_fMRIZ_soft = accMOE_fMRIZ_soft.T
print ('accMOE_fMRIZ_soft.shape:', accMOE_fMRIZ_soft.shape)
accMOE_fMRI_soft = np.zeros(shape=(1,n_folds))
accMOE_fMRI_soft = accMOE_fMRI_soft.T
print ('accMOE_fMRI_soft.shape:', accMOE_fMRI_soft.shape)


## sens
sensMOE_fMRIX_soft = np.zeros(shape=(1,n_folds))
sensMOE_fMRIX_soft = sensMOE_fMRIX_soft.T
print ('sensMOE_fMRIX_soft.shape:', sensMOE_fMRIX_soft.shape)
sensMOE_fMRIY_soft = np.zeros(shape=(1,n_folds))
sensMOE_fMRIY_soft = sensMOE_fMRIY_soft.T
print ('sensMOE_fMRIY_soft.shape:', sensMOE_fMRIY_soft.shape)
sensMOE_fMRIZ_soft = np.zeros(shape=(1,n_folds))
sensMOE_fMRIZ_soft = sensMOE_fMRIZ_soft.T
print ('sensMOE_fMRIZ_soft.shape:', sensMOE_fMRIZ_soft.shape)
sensMOE_fMRI_soft = np.zeros(shape=(1,n_folds))
sensMOE_fMRI_soft = sensMOE_fMRI_soft.T
print ('sensMOE_fMRI_soft.shape:', sensMOE_fMRI_soft.shape)


## spec
specMOE_fMRIX_soft = np.zeros(shape=(1,n_folds))
specMOE_fMRIX_soft = specMOE_fMRIX_soft.T
print ('specMOE_fMRIX_soft.shape:', specMOE_fMRIX_soft.shape)
specMOE_fMRIY_soft = np.zeros(shape=(1,n_folds))
specMOE_fMRIY_soft = specMOE_fMRIY_soft.T
print ('specMOE_fMRIY_soft.shape:', specMOE_fMRIY_soft.shape)
specMOE_fMRIZ_soft = np.zeros(shape=(1,n_folds))
specMOE_fMRIZ_soft = specMOE_fMRIZ_soft.T
print ('specMOE_fMRIZ_soft.shape:', specMOE_fMRIZ_soft.shape)
specMOE_fMRI_soft = np.zeros(shape=(1,n_folds))
specMOE_fMRI_soft = specMOE_fMRI_soft.T
print ('specMOE_fMRI_soft.shape:', specMOE_fMRI_soft.shape)


## f1
f1MOE_fMRIX_soft = np.zeros(shape=(1,n_folds))
f1MOE_fMRIX_soft = f1MOE_fMRIX_soft.T
print ('f1MOE_fMRIX_soft.shape:', f1MOE_fMRIX_soft.shape)
f1MOE_fMRIY_soft = np.zeros(shape=(1,n_folds))
f1MOE_fMRIY_soft = f1MOE_fMRIY_soft.T
print ('f1MOE_fMRIY_soft.shape:', f1MOE_fMRIY_soft.shape)
f1MOE_fMRIZ_soft = np.zeros(shape=(1,n_folds))
f1MOE_fMRIZ_soft = f1MOE_fMRIZ_soft.T
print ('f1MOE_fMRIZ_soft.shape:', f1MOE_fMRIZ_soft.shape)
f1MOE_fMRI_soft = np.zeros(shape=(1,n_folds))
f1MOE_fMRI_soft = f1MOE_fMRI_soft.T
print ('f1MOE_fMRI_soft.shape:', f1MOE_fMRI_soft.shape)

################################# start folds #################################
for i in range (n_folds):

    print ('Running Fold:',i+1,"/", n_folds)

    X_test1 = data1[:part_size1]
    Y_test1 = label1[:part_size1]
    print ('X_test1 shape:', X_test1.shape)
    print ('Y_test shape:',Y_test1.shape)
 
    X_train1 = data1[part_size1+1:]
    Y_train = label1[part_size1+1:]
    print ('X_train1 shape:',X_train1.shape)
    print ('Y_train shape:',Y_train.shape)    
    
    X_test2 = data2[:part_size2]
    Y_test1 = label2[:part_size2]
    print ('X_test2 shape:',X_test2.shape)
    print ('Y_test shape:',Y_test1.shape)
 
    X_train2 = data2[part_size2+1:]
    Y_train = label2[part_size2+1:]
    print ('X_train2 shape:',X_train2.shape)
    print ('Y_train shape:',Y_train.shape) 
    
    X_test3 = data3[:part_size3]
    Y_test1 = label3[:part_size3]
    print ('X_test3 shape:',X_test3.shape)
    print ('Y_test shape:',Y_test1.shape)
 
    X_train3 = data3[part_size3+1:]
    Y_train = label3[part_size3+1:]
    print ('X_train3 shape:',X_train3.shape)
    print ('Y_train shape:',Y_train.shape)
    
    if K.image_dim_ordering() == 'th':
        X_train1 = X_train1.reshape(X_train1.shape[0], 1,img_Y, img_Z)
        X_test1 = X_test1.reshape(X_test1.shape[0], 1, img_Y, img_Z)
        input_shape1 = (1, img_Y, img_Z)
        input_img1 = Input(shape=(1, img_Y, img_Z))  
    else:
        X_train1 = X_train1.reshape(X_train1.shape[0], img_Y, img_Z, 1)
        X_test1 = X_test1.reshape(X_test1.shape[0], img_Y, img_Z, 1)
        input_shape1 = ( img_Y, img_Z, 1)
        input_shape1_2 = ( img_Y*img_Z, 1)
        input_img1 = Input(shape=( img_Y, img_Z, 1))
    
    if K.image_dim_ordering() == 'th':
        X_train2 = X_train2.reshape(X_train2.shape[0], 1, img_X, img_Z)
        X_test2 = X_test2.reshape(X_test2.shape[0], 1, img_X, img_Z)
        input_shape2 = (1, img_X, img_Z)
        input_img2 = Input(shape=(1, img_X, img_Z)) 
    else:
        X_train2 = X_train2.reshape(X_train2.shape[0], img_X, img_Z, 1)
        X_test2 = X_test2.reshape(X_test2.shape[0], img_X, img_Z, 1)
        input_shape2 = ( img_X, img_Z, 1) 
        input_shape2_2 = ( img_X*img_Z, 1)
        input_img2 = Input(shape=( img_X, img_Z, 1))
        
    if K.image_dim_ordering() == 'th':
        X_train3 = X_train3.reshape(X_train3.shape[0], 1, img_X, img_Y)
        X_test3 = X_test3.reshape(X_test3.shape[0], 1, img_X, img_Y)
        input_shape3 = (1, img_X, img_Y)
        input_img3 = Input(shape=(1, img_X, img_Y)) 
    else:
        X_train3 = X_train3.reshape(X_train3.shape[0], img_X, img_Y, 1)
        X_test3 = X_test3.reshape(X_test3.shape[0], img_X, img_Y, 1)
        input_shape3 = ( img_X, img_Y, 1)
        input_shape3_2 = ( img_X*img_Y, 1)
        input_img3 = Input(shape=( img_X, img_Y, 1)) 
               
        print('X_train1 shape:', X_train1.shape)
        print('X_train2 shape:', X_train2.shape) 
        print('X_train3 shape:', X_train3.shape)
        
        # convert class vectors to binary class matrices
        Y_trainB = np_utils.to_categorical(Y_train, nb_classes)
        Y_testB = np_utils.to_categorical(Y_test1, nb_classes)     
         
        
###### fMRI X #######        
        feature_fMRIX1_layers = [
        (Convolution2D(nb_filters1, kernel_size[0], kernel_size[1], 
                         border_mode='valid', input_shape=input_shape1, activation='relu')),                       
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Convolution2D(nb_filters1, kernel_size[0], kernel_size[1], activation='relu')),      
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Convolution2D(nb_filters1, kernel_size[0], kernel_size[1], activation='relu')),      
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Flatten()), 
        ]
        
        classification_fMRIX1_layers = [
         Dense(300, activation='relu'),
         Dropout(0.5),
         Dense(nb_classes, W_regularizer=l2 (0.01), activation='softmax')
        ]
#############        
        feature_fMRIX2_layers = [
        (Convolution2D(nb_filters2, kernel_size[0], kernel_size[1], 
                         border_mode='valid', input_shape=input_shape1, activation='relu')),                       
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Convolution2D(nb_filters2, kernel_size[0], kernel_size[1], activation='relu')),      
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Convolution2D(nb_filters2, kernel_size[0], kernel_size[1], activation='relu')),      
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Flatten()), 
        ]
        
        classification_fMRIX2_layers = [
         Dense(400, activation='relu'),
         Dropout(0.5),
         Dense(nb_classes, W_regularizer=l2 (0.01), activation='softmax')
        ] 
#############        
        feature_fMRIX3_layers = [
        (Convolution2D(nb_filters3, kernel_size[0], kernel_size[1], 
                         border_mode='valid', input_shape=input_shape1, activation='relu')),                       
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Convolution2D(nb_filters3, kernel_size[0], kernel_size[1], activation='relu')),      
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Convolution2D(nb_filters3, kernel_size[0], kernel_size[1], activation='relu')),      
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Flatten()), 
        ]
        
        classification_fMRIX3_layers = [
         Dense(500, activation='relu'),
         Dropout(0.5),
         Dense(nb_classes, W_regularizer=l2 (0.01), activation='softmax')
        ]        
        
###### fMRI Y #######         
        feature_fMRIY1_layers = [
        (Convolution2D(nb_filters1, kernel_size[0], kernel_size[1], 
                         border_mode='valid', input_shape=input_shape2, activation='relu')),                       
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Convolution2D(nb_filters1, kernel_size[0], kernel_size[1], activation='relu')),      
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Convolution2D(nb_filters1, kernel_size[0], kernel_size[1], activation='relu')),      
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Flatten()), 
        ]
        
        classification_fMRIY1_layers = [
         Dense(300, activation='relu'),
         Dropout(0.5),
         Dense(nb_classes,W_regularizer=l2 (0.01), activation='softmax')
        ]
############# 
        feature_fMRIY2_layers = [
        (Convolution2D(nb_filters2, kernel_size[0], kernel_size[1], 
                         border_mode='valid', input_shape=input_shape2, activation='relu')),                       
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Convolution2D(nb_filters2, kernel_size[0], kernel_size[1], activation='relu')),      
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Convolution2D(nb_filters2, kernel_size[0], kernel_size[1], activation='relu')),      
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Flatten()), 
        ]
        
        classification_fMRIY2_layers = [
         Dense(400, activation='relu'),
         Dropout(0.5),
         Dense(nb_classes,W_regularizer=l2 (0.01), activation='softmax')
        ]
#############        
        feature_fMRIY3_layers = [
        (Convolution2D(nb_filters3, kernel_size[0], kernel_size[1], 
                         border_mode='valid', input_shape=input_shape2, activation='relu')),                       
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Convolution2D(nb_filters3, kernel_size[0], kernel_size[1], activation='relu')),      
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Convolution2D(nb_filters3, kernel_size[0], kernel_size[1], activation='relu')),      
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Flatten()), 
        ]
        
        classification_fMRIY3_layers = [
         Dense(500, activation='relu'),
         Dropout(0.5),
         Dense(nb_classes,W_regularizer=l2 (0.01), activation='softmax')
        ]
        
###### fMRI Z #######    
        feature_fMRIZ1_layers = [
        (Convolution2D(nb_filters1, kernel_size[0], kernel_size[1], 
                         border_mode='valid', input_shape=input_shape3, activation='relu')),                    
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Convolution2D(nb_filters1, kernel_size[0], kernel_size[1], activation='relu')),      
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Convolution2D(nb_filters1, kernel_size[0], kernel_size[1], activation='relu')),      
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Flatten()), 
        ]
        
        classification_fMRIZ1_layers = [
         Dense(300, activation='relu'),
         Dropout(0.5),
         Dense(nb_classes, W_regularizer=l2 (0.01), activation='softmax')
        ]
############# 
        feature_fMRIZ2_layers = [
        (Convolution2D(nb_filters2, kernel_size[0], kernel_size[1], 
                         border_mode='valid', input_shape=input_shape3, activation='relu')),                    
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Convolution2D(nb_filters2, kernel_size[0], kernel_size[1], activation='relu')),      
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Convolution2D(nb_filters2, kernel_size[0], kernel_size[1], activation='relu')),      
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Flatten()), 
        ]
        
        classification_fMRIZ2_layers = [
         Dense(400, activation='relu'),
         Dropout(0.5),
         Dense(nb_classes, W_regularizer=l2 (0.01), activation='softmax')
        ]
############# 
        feature_fMRIZ3_layers = [
        (Convolution2D(nb_filters3, kernel_size[0], kernel_size[1], 
                         border_mode='valid', input_shape=input_shape3, activation='relu')),                    
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Convolution2D(nb_filters3, kernel_size[0], kernel_size[1], activation='relu')),      
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Convolution2D(nb_filters3, kernel_size[0], kernel_size[1], activation='relu')),      
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Flatten()), 
        ]
        
        classification_fMRIZ3_layers = [
         Dense(500, activation='relu'),
         Dropout(0.5),
         Dense(nb_classes, W_regularizer=l2 (0.01), activation='softmax')
        ]
          
###### gate fMRIX #####        
        feature_gatefMRIX_layers = [
        (Convolution2D(nb_filters_g, kernel_size[0], kernel_size[1], 
                         border_mode='valid', input_shape=input_shape1, activation='relu')),                       
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Convolution2D(nb_filters_g, kernel_size[0], kernel_size[1], activation='relu')),      
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Convolution2D(nb_filters_g, kernel_size[0], kernel_size[1], activation='relu')),      
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Flatten()), 
        ]
        
        classification_gatefMRIX_layers = [
         Dense(300, activation='relu'),
         Dropout(0.5),
         Dense(nb_classes, W_regularizer=l2 (0.01), activation='softmax')
        ]

###### gate fMRIY #####          
        feature_gatefMRIY_layers = [
        (Convolution2D(nb_filters_g, kernel_size[0], kernel_size[1], 
                         border_mode='valid', input_shape=input_shape2, activation='relu')),                       
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Convolution2D(nb_filters_g, kernel_size[0], kernel_size[1], activation='relu')),      
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Convolution2D(nb_filters_g, kernel_size[0], kernel_size[1], activation='relu')),      
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Flatten()), 
        ]
        
        classification_gatefMRIY_layers = [
         Dense(300, activation='relu'),
         Dropout(0.5),
         Dense(nb_classes, W_regularizer=l2 (0.01), activation='softmax')
        ]        
###### gate fMRIZ #####    
        feature_gatefMRIZ_layers = [
        (Convolution2D(nb_filters_g, kernel_size[0], kernel_size[1], 
                         border_mode='valid', input_shape=input_shape3, activation='relu')),                    
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Convolution2D(nb_filters_g, kernel_size[0], kernel_size[1], activation='relu')),      
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Convolution2D(nb_filters_g, kernel_size[0], kernel_size[1], activation='relu')),      
        (MaxPooling2D(pool_size=pool_size)),
        #(Dropout(0.5)),
        (Flatten()), 
        ]
        
        classification_gatefMRIZ_layers = [
        #(Dense(100, activation='relu'))
         Dense(300, activation='relu'),
         Dropout(0.5),
         Dense(nb_classes, W_regularizer=l2 (0.01), activation='softmax')
        ] 
        
#############                 
############# fMRI #############               
        modelfMRIX1 = Sequential(feature_fMRIX1_layers + classification_fMRIX1_layers )
        modelfMRIX2 = Sequential(feature_fMRIX2_layers + classification_fMRIX2_layers )
        modelfMRIX3 = Sequential(feature_fMRIX3_layers + classification_fMRIX3_layers )
        
        modelfMRIY1 = Sequential(feature_fMRIY1_layers + classification_fMRIY1_layers)
        modelfMRIY2 = Sequential(feature_fMRIY2_layers + classification_fMRIY2_layers)
        modelfMRIY3 = Sequential(feature_fMRIY3_layers + classification_fMRIY3_layers)
        
        modelfMRIZ1 = Sequential(feature_fMRIZ1_layers + classification_fMRIZ1_layers)
        modelfMRIZ2 = Sequential(feature_fMRIZ2_layers + classification_fMRIZ2_layers)
        modelfMRIZ3 = Sequential(feature_fMRIZ3_layers + classification_fMRIZ3_layers)

        gatefMRIX = Sequential(feature_gatefMRIX_layers + classification_gatefMRIX_layers )
        gatefMRIY = Sequential(feature_gatefMRIY_layers + classification_gatefMRIY_layers)
        gatefMRIZ = Sequential(feature_gatefMRIZ_layers + classification_gatefMRIZ_layers)
########################################### 
### fMRI ###
        modelfMRIX1.load_weights('CNNX_Weight_ABIDEI_Un10.h5')   ## Filter = 8 , Dense = 300
        modelfMRIX2.load_weights('CNNX_Weight_ABIDEI_Un10_Filter10_Dense400.h5')
        modelfMRIX3.load_weights('CNNX_Weight_ABIDEI_Un10_Filter12_Dense500.h5')
        gatefMRIX.load_weights('CNNX_Weight_ABIDEI_Un10.h5')     ## Filter = 8 , Dense = 300

        modelfMRIY1.load_weights('CNNY_Weight_ABIDEI_Un10.h5')   ## Filter = 8 , Dense = 300
        modelfMRIY2.load_weights('CNNY_Weight_ABIDEI_Un10_Filter10_Dense400.h5')
        modelfMRIY3.load_weights('CNNY_Weight_ABIDEI_Un10_Filter12_Dense500.h5')
        gatefMRIY.load_weights('CNNY_Weight_ABIDEI_Un10.h5')     ## Filter = 8 , Dense = 300
        
        modelfMRIZ1.load_weights('CNNZ_Weight_ABIDEI_Un10.h5')   ## Filter = 8 , Dense = 300
        modelfMRIZ2.load_weights('CNNZ_Weight_ABIDEI_Un10_Filter10_Dense400.h5')
        modelfMRIZ3.load_weights('CNNZ_Weight_ABIDEI_Un10_Filter12_Dense500.h5')
        gatefMRIZ.load_weights('CNNZ_Weight_ABIDEI_Un10.h5')     ## Filter = 8 , Dense = 300 
                                    
## pop last layer and insert my own        
        gatefMRIX.layers.pop()
        gatefMRIX.add(Dropout(0.5))
        gatefMRIX.add(Dense(nb_output_g, W_regularizer=l2 (0.01), activation='softmax'))
        
        gatefMRIY.layers.pop()
        gatefMRIY.add(Dropout(0.5))
        gatefMRIY.add(Dense(nb_output_g, W_regularizer=l2 (0.01), activation='softmax'))
        
        gatefMRIZ.layers.pop()
        gatefMRIZ.add(Dropout(0.5))
        gatefMRIZ.add(Dense(nb_output_g, W_regularizer=l2 (0.01), activation='softmax'))
        
                  
########################################### 
############# fMRI #############        
        def merge_modelfMRIX(branches):
            gfMRIX, ofMRIX1, ofMRIX2, ofMRIX3 = branches
            return K.transpose(K.transpose(ofMRIX1) * gfMRIX[:, 0]+ K.transpose(ofMRIX2) * gfMRIX[:, 1]
                                                                  + K.transpose(ofMRIX3) * gfMRIX[:, 2])            
        def merge_modelfMRIY(branches):
            gfMRIY, ofMRIY1, ofMRIY2, ofMRIY3 = branches
            return K.transpose(K.transpose(ofMRIY1) * gfMRIY[:, 0]+ K.transpose(ofMRIY2) * gfMRIY[:, 1]
                                                                  + K.transpose(ofMRIY3) * gfMRIY[:, 2])
        def merge_modelfMRIZ(branches):
            gfMRIZ, ofMRIZ1, ofMRIZ2, ofMRIZ3 = branches
            return K.transpose(K.transpose(ofMRIZ1) * gfMRIZ[:, 0]+ K.transpose(ofMRIZ2) * gfMRIZ[:, 1]
                                                                  + K.transpose(ofMRIZ3) * gfMRIZ[:, 2]) 
                                                                                                                                  
###############################################################################
############# fMRI ############# 
        MOE_fMRIX = Sequential()
        MOE_fMRIX.add(Merge([gatefMRIX, modelfMRIX1, modelfMRIX2, modelfMRIX3], output_shape=(2,), mode=merge_modelfMRIX))
        print ('MOE_fMRIX.output_shape:')
        print (MOE_fMRIX.output_shape)
        
        MOE_fMRIY = Sequential()
        MOE_fMRIY.add(Merge([gatefMRIY, modelfMRIY1, modelfMRIY2, modelfMRIY3], output_shape=(2,), mode=merge_modelfMRIY))
        print ('MOE_fMRIY.output_shape:')
        print (MOE_fMRIY.output_shape) 
        
        MOE_fMRIZ = Sequential()
        MOE_fMRIZ.add(Merge([gatefMRIZ, modelfMRIZ1, modelfMRIZ2, modelfMRIZ3], output_shape=(2,), mode=merge_modelfMRIZ))
        print ('MOE_fMRIZ.output_shape:')
        print (MOE_fMRIZ.output_shape)                             
###############################################################################  
############# fMRI #############         
        MOE_fMRIX.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        MOE_fMRIY.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        MOE_fMRIZ.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])       
############################### Freeze features layers ########################  
############# fMRI #############         
        for l_fMRIX1 in  feature_fMRIX1_layers:
            l_fMRIX1.trainable = False
            
        for l_fMRIX2 in  feature_fMRIX2_layers:
            l_fMRIX2.trainable = False  
            
        for l_fMRIX3 in  feature_fMRIX3_layers:
            l_fMRIX3.trainable = False            

################              
        for l_fMRIY1 in  feature_fMRIY1_layers:
            l_fMRIY1.trainable = False 
            
        for l_fMRIY2 in  feature_fMRIY2_layers:
            l_fMRIY2.trainable = False  
            
        for l_fMRIY3 in  feature_fMRIY3_layers:
            l_fMRIY3.trainable = False             

################          
        for l_fMRIZ1 in  feature_fMRIZ1_layers:
            l_fMRIZ1.trainable = False 
            
        for l_fMRIZ2 in  feature_fMRIZ2_layers:
            l_fMRIZ2.trainable = False  
            
        for l_fMRIZ3 in  feature_fMRIZ3_layers:
            l_fMRIZ3.trainable = False             

################   
        for l_gatefMRIX in  feature_gatefMRIX_layers:
            l_gatefMRIX.trainable = False
            
        for l_gatefMRIY in  feature_gatefMRIY_layers:
            l_gatefMRIY.trainable = False  
        
        for l_gatefMRIZ in  feature_gatefMRIZ_layers:
            l_gatefMRIZ.trainable = False 

###############################################################################  
############# fMRI ############# 
        
        MOE_fMRIX.fit([X_train1,X_train1,X_train1,X_train1], Y_trainB, 
                batch_size=batch_size, nb_epoch=nb_epoch, validation_data=([X_test1,X_test1,X_test1,X_test1], Y_testB)
                                       , shuffle=True) 

                
        MOE_fMRIY.fit([X_train2,X_train2,X_train2,X_train2], Y_trainB, 
                batch_size=batch_size, nb_epoch=nb_epoch, validation_data=([X_test2,X_test2,X_test2,X_test2], Y_testB)
                                       , shuffle=True)
                                                       
                
        MOE_fMRIZ.fit([X_train3,X_train3,X_train3,X_train3], Y_trainB, 
                batch_size=batch_size, nb_epoch=nb_epoch, validation_data=([X_test3,X_test3,X_test3,X_test3], Y_testB)
                                       , shuffle=True)
                                      
###############################################################################
############# fMRI #############                 
        scoreMOE_fMRIX_soft = MOE_fMRIX.evaluate([X_test1,X_test1,X_test1,X_test1], Y_testB, verbose=0)        
        scoreMOE_fMRIY_soft = MOE_fMRIY.evaluate([X_test2,X_test2,X_test2,X_test2], Y_testB, verbose=0) 
        scoreMOE_fMRIZ_soft = MOE_fMRIZ.evaluate([X_test3,X_test3,X_test3,X_test3], Y_testB, verbose=0)  
         
        predsMOE_fMRIX_soft = MOE_fMRIX.predict_classes([X_test1,X_test1,X_test1,X_test1])
        predsMOE_fMRIY_soft = MOE_fMRIY.predict_classes([X_test2,X_test2,X_test2,X_test2])
        predsMOE_fMRIZ_soft = MOE_fMRIZ.predict_classes([X_test3,X_test3,X_test3,X_test3]) 

               
        cmMOE_fMRIX_soft = confusion_matrix(Y_test1, predsMOE_fMRIX_soft)     
        sensitivityMOE_fMRIX_soft = cmMOE_fMRIX_soft[0][0] / (cmMOE_fMRIX_soft[0][0] + cmMOE_fMRIX_soft[0][1])  
        specificityMOE_fMRIX_soft = cmMOE_fMRIX_soft[1][1] / (cmMOE_fMRIX_soft[1][1] + cmMOE_fMRIX_soft[1][0])
        fmeasureMOE_fMRIX_soft = 2 * (specificityMOE_fMRIX_soft * sensitivityMOE_fMRIX_soft) / (specificityMOE_fMRIX_soft + sensitivityMOE_fMRIX_soft)

        cmMOE_fMRIY_soft = confusion_matrix(Y_test1, predsMOE_fMRIY_soft)     
        sensitivityMOE_fMRIY_soft = cmMOE_fMRIY_soft[0][0] / (cmMOE_fMRIY_soft[0][0] + cmMOE_fMRIY_soft[0][1])  
        specificityMOE_fMRIY_soft = cmMOE_fMRIY_soft[1][1] / (cmMOE_fMRIY_soft[1][1] + cmMOE_fMRIY_soft[1][0])
        fmeasureMOE_fMRIY_soft = 2 * (specificityMOE_fMRIY_soft * sensitivityMOE_fMRIY_soft) / (specificityMOE_fMRIY_soft + sensitivityMOE_fMRIY_soft)
        
        cmMOE_fMRIZ_soft = confusion_matrix(Y_test1, predsMOE_fMRIZ_soft)     
        sensitivityMOE_fMRIZ_soft = cmMOE_fMRIZ_soft[0][0] / (cmMOE_fMRIZ_soft[0][0] + cmMOE_fMRIZ_soft[0][1])  
        specificityMOE_fMRIZ_soft = cmMOE_fMRIZ_soft[1][1] / (cmMOE_fMRIZ_soft[1][1] + cmMOE_fMRIZ_soft[1][0])
        fmeasureMOE_fMRIZ_soft = 2 * (specificityMOE_fMRIZ_soft * sensitivityMOE_fMRIZ_soft) / (specificityMOE_fMRIZ_soft + sensitivityMOE_fMRIZ_soft)


        preds1 = np.vstack((predsMOE_fMRIX_soft,predsMOE_fMRIY_soft))
        predsMOE_fMRI_soft = np.vstack((preds1,predsMOE_fMRIZ_soft))        
###############################################################################    
############################# preds on Train Data #############################  
        predsMOE_fMRIX_TrainSoft = MOE_fMRIX.predict_classes([X_train1,X_train1,X_train1,X_train1])        
        predsMOE_fMRIY_TrainSoft = MOE_fMRIY.predict_classes([X_train2,X_train2,X_train2,X_train2])        
        predsMOE_fMRIZ_TrainSoft = MOE_fMRIZ.predict_classes([X_train3,X_train3,X_train3,X_train3]) 
################################ CM on Train Data ############################## 
        cmfMRIX_TrainSoft = confusion_matrix(Y_train, predsMOE_fMRIX_TrainSoft)                     
        cmfMRIY_TrainSoft = confusion_matrix(Y_train, predsMOE_fMRIY_TrainSoft)             
        cmfMRIZ_TrainSoft = confusion_matrix(Y_train, predsMOE_fMRIZ_TrainSoft)  
############################   LM on Train Data   #############################              
        ## LM
        ## LM fMRIX      
        lmfMRIX_soft = np.zeros(shape=(2,2))
        col1 = cmfMRIX_TrainSoft[0,0]+cmfMRIX_TrainSoft[1,0]
        print ('col1:')
        print (col1)
        col2 = cmfMRIX_TrainSoft[0,1]+cmfMRIX_TrainSoft[1,1]
        print ('col2:')
        print (col2)

        lmfMRIX_soft[0,0] = cmfMRIX_TrainSoft[0,0]/col1
        lmfMRIX_soft[1,0] = cmfMRIX_TrainSoft[1,0]/col1

        lmfMRIX_soft[0,1] = cmfMRIX_TrainSoft[0,1]/col2
        lmfMRIX_soft[1,1] = cmfMRIX_TrainSoft[1,1]/col2

        print ('lmfMRIX_TrainSoft Matrix:')
        print (lmfMRIX_soft)
        ## LM fMRIY      
        lmfMRIY_soft = np.zeros(shape=(2,2))
        col1 = cmfMRIY_TrainSoft[0,0]+cmfMRIY_TrainSoft[1,0]
        print ('col1:')
        print (col1)
        col2 = cmfMRIY_TrainSoft[0,1]+cmfMRIY_TrainSoft[1,1]
        print ('col2:')
        print (col2)

        lmfMRIY_soft[0,0] = cmfMRIY_TrainSoft[0,0]/col1
        lmfMRIY_soft[1,0] = cmfMRIY_TrainSoft[1,0]/col1

        lmfMRIY_soft[0,1] = cmfMRIY_TrainSoft[0,1]/col2
        lmfMRIY_soft[1,1] = cmfMRIY_TrainSoft[1,1]/col2

        print ('lmfMRIY_TrainSoft Matrix:')
        print (lmfMRIY_soft)
        ## LM fMRIZ      
        lmfMRIZ_soft = np.zeros(shape=(2,2))
        col1 = cmfMRIZ_TrainSoft[0,0]+cmfMRIZ_TrainSoft[1,0]
        print ('col1:')
        print (col1)
        col2 = cmfMRIZ_TrainSoft[0,1]+cmfMRIZ_TrainSoft[1,1]
        print ('col2:')
        print (col2)

        lmfMRIZ_soft[0,0] = cmfMRIZ_TrainSoft[0,0]/col1
        lmfMRIZ_soft[1,0] = cmfMRIZ_TrainSoft[1,0]/col1

        lmfMRIZ_soft[0,1] = cmfMRIZ_TrainSoft[0,1]/col2
        lmfMRIZ_soft[1,1] = cmfMRIZ_TrainSoft[1,1]/col2

        print ('lmfMRIZ_TrainSoft Matrix:')
        print (lmfMRIZ_soft)              
################        
################################## fMRI #######################################
        Mu0_fMRI = np.zeros(shape=(1,X_test1.shape[0]))
        Mu1_fMRI = np.zeros(shape=(1,X_test1.shape[0]))
        
        for i_fMRI in range(X_test1.shape[0]):
            Mu0_fMRI[0,i_fMRI] = lmfMRIX_soft[0,predsMOE_fMRI_soft[0,i_fMRI]]+lmfMRIY_soft[0,predsMOE_fMRI_soft[1,i_fMRI]]+lmfMRIZ_soft[0,predsMOE_fMRI_soft[2,i_fMRI]]
            Mu1_fMRI[0,i_fMRI] = lmfMRIX_soft[1,predsMOE_fMRI_soft[0,i_fMRI]]+lmfMRIY_soft[1,predsMOE_fMRI_soft[1,i_fMRI]]+lmfMRIZ_soft[1,predsMOE_fMRI_soft[2,i_fMRI]]        

        predsfMRI_soft = np.zeros(shape=(1,X_test1.shape[0]))

        for j_fMRI in range(X_test1.shape[0]):
            if Mu0_fMRI[0,j_fMRI] > Mu1_fMRI[0,j_fMRI]:
               predsfMRI_soft[0,j_fMRI] = 0
            elif Mu0_fMRI[0,j_fMRI] < Mu1_fMRI[0,j_fMRI]:
               predsfMRI_soft[0,j_fMRI] = 1        
                
        predsfMRI_soft = predsfMRI_soft.T
        ## CM on Test Data
        cm = confusion_matrix(Y_test1, predsfMRI_soft) 
        print ('confusion matrix:')
        print(cm)
        ##
        sc_fMRI = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
        sensitivity_fMRI = cm[0][0] / (cm[0][0] + cm[0][1])  
        specificity_fMRI = cm[1][1] / (cm[1][1] + cm[1][0])
        fmeasure_fMRI = 2 * (specificity_fMRI * sensitivity_fMRI) / (specificity_fMRI + sensitivity_fMRI)  
        
############################################################################### 
################ fMRI #############        
        ##acc       
        accMOE_fMRIX_soft[i] = scoreMOE_fMRIX_soft[1]
        print ('accMOE_fMRIX_soft:', accMOE_fMRIX_soft[i])
        
        accMOE_fMRIY_soft[i] = scoreMOE_fMRIY_soft[1]
        print ('accMOE_fMRIY_soft:', accMOE_fMRIY_soft[i]) 
        
        accMOE_fMRIZ_soft[i] = scoreMOE_fMRIZ_soft[1]
        print ('accMOE_fMRIZ_soft:', accMOE_fMRIZ_soft[i])
        
        accMOE_fMRI_soft[i] = sc_fMRI
        print ('accMOE_fMRI_soft:', accMOE_fMRI_soft[i])   
                        

        ##sens       
        sensMOE_fMRIX_soft[i] = sensitivityMOE_fMRIX_soft
        print ('sensMOE_fMRIX_soft:', sensMOE_fMRIX_soft[i])
        
        sensMOE_fMRIY_soft[i] = sensitivityMOE_fMRIY_soft
        print ('sensMOE_fMRIY_soft:', sensMOE_fMRIY_soft[i]) 
        
        sensMOE_fMRIZ_soft[i] = sensitivityMOE_fMRIZ_soft
        print ('sensMOE_fMRIZ_soft:', sensMOE_fMRIZ_soft[i]) 
        
        sensMOE_fMRI_soft[i] = sensitivity_fMRI
        print ('sensMOE_fMRI_soft:', sensMOE_fMRI_soft[i])
               

        ##spec       
        specMOE_fMRIX_soft[i] = specificityMOE_fMRIX_soft
        print ('specMOE_fMRIX_soft:', specMOE_fMRIX_soft[i])
        
        specMOE_fMRIY_soft[i] = specificityMOE_fMRIY_soft
        print ('specMOE_fMRIY_soft:', specMOE_fMRIY_soft[i]) 
        
        specMOE_fMRIZ_soft[i] = specificityMOE_fMRIZ_soft
        print ('specMOE_fMRIZ_soft:', specMOE_fMRIZ_soft[i]) 
        
        specMOE_fMRI_soft[i] = specificity_fMRI
        print ('specMOE_fMRI_soft:', specMOE_fMRI_soft[i])
        
       
        ##f1       
        f1MOE_fMRIX_soft[i] = fmeasureMOE_fMRIX_soft
        print ('f1MOE_fMRIX_soft:', f1MOE_fMRIX_soft[i])
        
        f1MOE_fMRIY_soft[i] = fmeasureMOE_fMRIY_soft
        print ('f1MOE_fMRIY_soft:', f1MOE_fMRIY_soft[i]) 
        
        f1MOE_fMRIZ_soft[i] = fmeasureMOE_fMRIZ_soft
        print ('f1MOE_fMRIZ_soft:', f1MOE_fMRIZ_soft[i]) 
        
        f1MOE_fMRI_soft[i] = fmeasure_fMRI
        print ('f1MOE_fMRI_soft:', f1MOE_fMRI_soft[i]) 
###############################################################################        
        data1 = np.roll(data1, part_size1, axis=0)
        label1 = np.roll(label1, part_size1, axis=0)
    
        data2 = np.roll(data2, part_size2, axis=0)
        label2 = np.roll(label2, part_size2, axis=0)
    
        data3 = np.roll(data3, part_size3, axis=0)
        label3 = np.roll(label3, part_size3, axis=0)        
############################### softmax results ###############################
## acc        
acc_fMRIX_soft = (np.sum(accMOE_fMRIX_soft))/n_folds
print ('accuracy average softmax (MOE_fMRIX):', acc_fMRIX_soft)
acc_fMRIY_soft = (np.sum(accMOE_fMRIY_soft))/n_folds
print ('accuracy average softmax (MOE_fMRIY):', acc_fMRIY_soft)
acc_fMRIZ_soft = (np.sum(accMOE_fMRIZ_soft))/n_folds
print ('accuracy average softmax (MOE_fMRIZ):', acc_fMRIZ_soft)
acc_fMRI_soft = (np.sum(accMOE_fMRI_soft))/n_folds
print ('accuracy average softmax (MOE_fMRI):', acc_fMRI_soft)

STDacc_fMRIX_soft = (np.std(accMOE_fMRIX_soft))
print ('STD accuracy average softmax (MOE_fMRIX):', STDacc_fMRIX_soft)
STDacc_fMRIY_soft = (np.std(accMOE_fMRIY_soft))
print ('STD accuracy average softmax (MOE_fMRIY):', STDacc_fMRIY_soft)
STDacc_fMRIZ_soft = (np.std(accMOE_fMRIZ_soft))
print ('STD accuracy average softmax (MOE_fMRIZ):', STDacc_fMRIZ_soft)
STDacc_fMRI_soft = (np.std(accMOE_fMRI_soft))
print ('STD accuracy average softmax (MOE_fMRI):', STDacc_fMRI_soft)

## sens        
sens_fMRIX_soft = (np.sum(sensMOE_fMRIX_soft))/n_folds
print ('sensitivity average softmax (MOE_fMRIX):', sens_fMRIX_soft)
sens_fMRIY_soft = (np.sum(sensMOE_fMRIY_soft))/n_folds
print ('sensitivity average softmax (MOE_fMRIY):', sens_fMRIY_soft)
sens_fMRIZ_soft = (np.sum(sensMOE_fMRIZ_soft))/n_folds
print ('sensitivity average softmax (MOE_fMRIZ):', sens_fMRIZ_soft)
sens_fMRI_soft = (np.sum(sensMOE_fMRI_soft))/n_folds
print ('sensitivity average softmax (MOE_fMRI):', sens_fMRI_soft)

STDsens_fMRIX_soft = (np.std(sensMOE_fMRIX_soft))
print ('STD sensitivity average softmax (MOE_fMRIX):', STDsens_fMRIX_soft)
STDsens_fMRIY_soft = (np.std(sensMOE_fMRIY_soft))
print ('STD sensitivity average softmax (MOE_fMRIY):', STDsens_fMRIY_soft)
STDsens_fMRIZ_soft = (np.std(sensMOE_fMRIZ_soft))
print ('STD sensitivity average softmax (MOE_fMRIZ):', STDsens_fMRIZ_soft)
STDsens_fMRI_soft = (np.std(sensMOE_fMRI_soft))
print ('STD sensitivity average softmax (MOE_fMRI):', STDsens_fMRI_soft)


## spec        
spec_fMRIX_soft = (np.sum(specMOE_fMRIX_soft))/n_folds
print ('specificity average softmax (MOE_fMRIX):', spec_fMRIX_soft)
spec_fMRIY_soft = (np.sum(specMOE_fMRIY_soft))/n_folds
print ('specificity average softmax (MOE_fMRIY):', spec_fMRIY_soft)
spec_fMRIZ_soft = (np.sum(specMOE_fMRIZ_soft))/n_folds
print ('specificity average softmax (MOE_fMRIZ):', spec_fMRIZ_soft)
spec_fMRI_soft = (np.sum(specMOE_fMRI_soft))/n_folds
print ('specificity average softmax (MOE_fMRI):', spec_fMRI_soft)

STDspec_fMRIX_soft = (np.std(specMOE_fMRIX_soft))
print ('STD specificity average softmax (MOE_fMRIX):', STDspec_fMRIX_soft)
STDspec_fMRIY_soft = (np.std(specMOE_fMRIY_soft))
print ('STD specificity average softmax (MOE_fMRIY):', STDspec_fMRIY_soft)
STDspec_fMRIZ_soft = (np.std(specMOE_fMRIZ_soft))
print ('STD specificity average softmax (MOE_fMRIZ):', STDspec_fMRIZ_soft)
STDspec_fMRI_soft = (np.std(specMOE_fMRI_soft))
print ('STD specificity average softmax (MOE_fMRI):', STDspec_fMRI_soft)


## f1        
f1MOE_fMRIX_soft = (np.sum(f1MOE_fMRIX_soft))/n_folds
print ('f1-score average softmax (MOE_fMRIX):', f1MOE_fMRIX_soft)
f1MOE_fMRIY_soft = (np.sum(f1MOE_fMRIY_soft))/n_folds
print ('f1-score average softmax (MOE_fMRIY):', f1MOE_fMRIY_soft)
f1MOE_fMRIZ_soft = (np.sum(f1MOE_fMRIZ_soft))/n_folds
print ('f1-score average softmax (MOE_fMRIZ):', f1MOE_fMRIZ_soft)
f1MOE_fMRI_soft = (np.sum(f1MOE_fMRI_soft))/n_folds
print ('f1-score average softmax (MOE_fMRI):', f1MOE_fMRI_soft)

STDf1_fMRIX_soft = (np.std(f1MOE_fMRIX_soft))
print ('STD f1-score average softmax (MOE_fMRIX):', STDf1_fMRIX_soft)
STDf1_fMRIY_soft = (np.std(f1MOE_fMRIY_soft))
print ('STD f1-score average softmax (MOE_fMRIY):', STDf1_fMRIY_soft)
STDf1_fMRIZ_soft = (np.std(f1MOE_fMRIZ_soft))
print ('STD f1-score average softmax (MOE_fMRIZ):', STDf1_fMRIZ_soft)
STDf1_fMRI_soft = (np.std(f1MOE_fMRI_soft))
print ('STD f1-score average softmax (MOE_fMRI):', STDf1_fMRI_soft)

##### Ttest #####
from scipy import stats
ttest_acc_X, pval_acc_X = stats.ttest_rel(accMOE_fMRIX_soft,accMOE_fMRI_soft)
ttest_acc_Y, pval_acc_Y = stats.ttest_rel(accMOE_fMRIY_soft,accMOE_fMRI_soft)       
ttest_acc_Z, pval_acc_Z = stats.ttest_rel(accMOE_fMRIZ_soft,accMOE_fMRI_soft)

print ('Ttest accuracy softmax (MOE_fMRIX):', ttest_acc_X)
print ('Ttest accuracy softmax (MOE_fMRIY):', ttest_acc_Y)
print ('Ttest accuracy softmax (MOE_fMRIZ):', ttest_acc_Z) 

print ('Pval accuracy softmax (MOE_fMRIX):', pval_acc_X)
print ('Pval accuracy softmax (MOE_fMRIY):', pval_acc_Y)
print ('Pval accuracy softmax (MOE_fMRIZ):', pval_acc_Z) 
####
ttest_sens_X, pval_sens_X = stats.ttest_rel(sensMOE_fMRIX_soft,sensMOE_fMRI_soft)
ttest_sens_Y, pval_sens_Y = stats.ttest_rel(sensMOE_fMRIY_soft,sensMOE_fMRI_soft)       
ttest_sens_Z, pval_sens_Z = stats.ttest_rel(sensMOE_fMRIZ_soft,sensMOE_fMRI_soft)

print ('Ttest sensuracy softmax (MOE_fMRIX):', ttest_sens_X)
print ('Ttest sensuracy softmax (MOE_fMRIY):', ttest_sens_Y)
print ('Ttest sensuracy softmax (MOE_fMRIZ):', ttest_sens_Z) 

print ('Pval sensuracy softmax (MOE_fMRIX):', pval_sens_X)
print ('Pval sensuracy softmax (MOE_fMRIY):', pval_sens_Y)
print ('Pval sensuracy softmax (MOE_fMRIZ):', pval_sens_Z) 
####
ttest_spec_X, pval_spec_X = stats.ttest_rel(specMOE_fMRIX_soft,specMOE_fMRI_soft)
ttest_spec_Y, pval_spec_Y = stats.ttest_rel(specMOE_fMRIY_soft,specMOE_fMRI_soft)       
ttest_spec_Z, pval_spec_Z = stats.ttest_rel(specMOE_fMRIZ_soft,specMOE_fMRI_soft)

print ('Ttest specuracy softmax (MOE_fMRIX):', ttest_spec_X)
print ('Ttest specuracy softmax (MOE_fMRIY):', ttest_spec_Y)
print ('Ttest specuracy softmax (MOE_fMRIZ):', ttest_spec_Z) 

print ('Pval specuracy softmax (MOE_fMRIX):', pval_spec_X)
print ('Pval specuracy softmax (MOE_fMRIY):', pval_spec_Y)
print ('Pval specuracy softmax (MOE_fMRIZ):', pval_spec_Z)
#####
print ('accMOE_fMRIX_soft:',accMOE_fMRIX_soft)
print ('sensMOE_fMRIX_soft:',sensMOE_fMRIX_soft)
print ('specMOE_fMRIX_soft:',specMOE_fMRIX_soft)

print ('accMOE_fMRIY_soft:',accMOE_fMRIY_soft)
print ('sensMOE_fMRIY_soft:',sensMOE_fMRIY_soft)
print ('specMOE_fMRIY_soft:',specMOE_fMRIY_soft)

print ('accMOE_fMRIZ_soft:',accMOE_fMRIZ_soft)
print ('sensMOE_fMRIZ_soft:',sensMOE_fMRIZ_soft)
print ('specMOE_fMRIZ_soft:',specMOE_fMRIZ_soft)

print ('accMOE_fMRI_soft:',accMOE_fMRI_soft)
print ('sensMOE_fMRI_soft:',sensMOE_fMRI_soft)
print ('specMOE_fMRI_soft:',specMOE_fMRI_soft)