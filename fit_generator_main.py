from fit_loader import IEMR
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense,AvgPool2D,DepthwiseConv2D,SeparableConv2D
from keras import Model
import numpy as np
from keras import backend as K
import argparse
from keras.models import Sequential
from keras.layers.merge import concatenate
gpus= tf.config.list_physical_devices('GPU')
print(gpus)
tf.config.experimental.set_memory_growth(gpus[0], True)
import torch
###############################
input = Input(shape=(12,100,1))
x = Conv2D(filters=16, kernel_size=(1,12), padding='valid', activation='relu')(input)
x = AvgPool2D(pool_size=2, strides=2, padding='valid')(x)


x = Conv2D(filters=32, kernel_size=(1,12), padding='same', activation='relu')(x)
x = AvgPool2D(pool_size=2, strides=2,padding='valid')(x)
x = Conv2D(filters=64, kernel_size=(1,12),padding='same',activation='relu')(x)
x = Conv2D(filters=64, kernel_size=(1,12),padding='same',activation='relu')(x)

x = DepthwiseConv2D(kernel_size=(1,12),depth_multiplier=2)(x)
x = SeparableConv2D(filters=64, kernel_size=1,depth_multiplier=1)(x)
out = Flatten()(x)
window_output= Dense(units=100, activation='softmax')(out)
signal_output= Dense(units=12, activation='softmax')(out)
#output = Dense(units=12, activation='softmax')(x)

output=[window_output,signal_output]
#################################

def custom_loss(y_true, y_pred):
            
    # calculate loss, using y_pred
    #print(y_true)
    #print(y_pred[0],y_pred[1])
    a=y_true[0:99]
    b=y_true[100:111]
    #print(a,b)
    g1=K.categorical_crossentropy(target=a,output=y_pred[0])
    g2=K.categorical_crossentropy(target=b,output=y_pred[2])   
    log_loss=g1+g2

    print(log_loss)        
    return log_loss

if __name__=="__main__":
     parser = argparse.ArgumentParser(description='Process some integers.')
     parser.add_argument('--downstream', default=None)
     args = parser.parse_args()

     #model=Model(inputs=input, outputs=output)
     
     model1 = Model(inputs=input, outputs=window_output)
     model2 = Model(inputs=input, outputs=signal_output)
     concatenated = concatenate([window_output, signal_output])
     model = Model(inputs=input, outputs=concatenated)   
     #model.summary()
     training_generator=IEMR()
     count=0
     for data,labels in training_generator:
       count=count+1
       print(count)
  
     model.compile(optimizer='adam',loss= 'categorical_crossentropy',metrics=['accuracy'])

     model.fit_generator(training_generator,steps_per_epoch=2200,epochs=100)   
    

   
     