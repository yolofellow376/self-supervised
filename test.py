from myloader import IEMR
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense,AvgPool2D,DepthwiseConv2D,SeparableConv2D
from tensorflow.keras import Model
import numpy as np
import argparse
input = Input(shape=(12,100,1))
from keras import backend as K
from keras.models import Sequential

x = Conv2D(filters=16, kernel_size=(1,12), padding='valid', activation='relu')(input)
x = AvgPool2D(pool_size=2, strides=2, padding='valid')(x)


x = Conv2D(filters=32, kernel_size=(1,12), padding='same', activation='relu')(x)
x = AvgPool2D(pool_size=2, strides=2,padding='valid')(x)
x = Conv2D(filters=64, kernel_size=(1,12),padding='same',activation='relu')(x)
x = Conv2D(filters=64, kernel_size=(1,12),padding='same',activation='relu')(x)

x = DepthwiseConv2D(kernel_size=(1,12),depth_multiplier=2)(x)
x = SeparableConv2D(filters=64, kernel_size=1,depth_multiplier=1)(x)
x = Flatten()(x)
window_output= Dense(units=100, activation='softmax')(x)
signal_output= Dense(units=12, activation='softmax')(x)
#output = Dense(units=12, activation='softmax')(x)
output=[window_output,signal_output]
model = Model(inputs=input, outputs=output)
arr=np.asarray(model.trainable_variables)
a=Sequential()
a.add(model)
a.summary()


#dy_dx = g.gradient(y,x)
#dy_dx = g.gradient(x,matrix)


print(args.downstream)