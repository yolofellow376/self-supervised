from one_label_fit_loader import IEMR
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense,AvgPool2D,DepthwiseConv2D,SeparableConv2D
from keras import Model
import numpy as np
from keras import backend as K
import argparse
from keras.models import Sequential
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping,ModelCheckpoint, ReduceLROnPlateau
import keras
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
window_output= Dense(units=30, activation='softmax')(out)
signal_output= Dense(units=30, activation='softmax')(out)
#output = Dense(units=12, activation='softmax')(x)

output=[window_output,signal_output]
#################################



if __name__=="__main__":
     parser = argparse.ArgumentParser(description='Process some integers.')
     parser.add_argument('--downstream', default=None)
     args = parser.parse_args()
     model = Model(inputs=input, outputs=[window_output,signal_output])   
     model.load_weights("./0428_data2200.hdf5")
     model.summary()
     if args.downstream=='RF':
       print('random forest')
       new_model=Sequential()
       #model = keras.models.load_model("./mdl_wts.hdf5")
       for layer in model.layers[:-2]:
          layer.trainable = False
          new_model.add(layer)
       model=new_model
       model.add(Dense(output_dim = 256, activation = 'relu'))
       model.add(Dense(output_dim = 128, activation = 'relu'))
       model.add(Dense(output_dim = 1, activation = 'relu'))   
     elif args.downstream=='3FC':
       print('3FC')
       new_model=Sequential()
       #model = keras.models.load_model("./mdl_wts.hdf5")
       for layer in model.layers[:-2]:
          layer.trainable = False
          new_model.add(layer)
       model=new_model
       model.add(Dense(output_dim = 256, activation = 'relu'))
       model.add(Dense(output_dim = 128, activation = 'relu'))
       model.add(Dense(output_dim = 1, activation = 'relu'))   
     model.summary()
     
     
    
   
     