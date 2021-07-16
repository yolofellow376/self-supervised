from clean_ver_loader import IEMR
from clean_ver_valid_loader import IEMR_valid
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

tf.compat.v1.enable_eager_execution(
    config=None,
    device_policy=None,
    execution_mode=None
)
import sys
gpus= tf.config.list_physical_devices('GPU')
print(gpus)
tf.config.experimental.set_memory_growth(gpus[0], True)
import torch
###############################
input = Input(shape=(12,1500,1))
x = Conv2D(filters=16, kernel_size=(1,12), padding='valid', activation='relu')(input)
x = AvgPool2D(pool_size=2, strides=2, padding='valid')(x)


x = Conv2D(filters=32, kernel_size=(1,12), padding='same', activation='relu')(x)
x = AvgPool2D(pool_size=2, strides=2,padding='valid')(x)
x = Conv2D(filters=64, kernel_size=(1,12),padding='same',activation='relu')(x)
x = Conv2D(filters=64, kernel_size=(1,12),padding='same',activation='relu')(x)

x = DepthwiseConv2D(kernel_size=(1,12),depth_multiplier=2)(x)
x = SeparableConv2D(filters=64, kernel_size=1,depth_multiplier=1)(x)
out = Flatten()(x)
window_output= Dense(units=500, activation='softmax')(out)
signal_output= Dense(units=10, activation='softmax')(out)
#output = Dense(units=12, activation='softmax')(x)

output=[window_output,signal_output]
#################################
f=open('what.txt','a')
def my_metric_fn(y_true, y_pred):

    f=open('what.txt','a')
    squared_difference = tf.square(y_true - y_pred)
    f.write(str(y_true)+'\n')
    f.write(str(y_pred)+'\n')
    f.close()
    #return tf.reduce_mean(squared_difference, axis=-1)
    return 10
if __name__=="__main__":
     parser = argparse.ArgumentParser(description='Process some integers.')
     parser.add_argument('--downstream', default=None)
     args = parser.parse_args()

     #model=Model(inputs=input, outputs=output)
     
     #model1 = Model(inputs=input, outputs=window_output)
     #model2 = Model(inputs=input, outputs=signal_output)
     #concatenated = concatenate([window_output, signal_output])
     model = Model(inputs=input, outputs=[window_output,signal_output])   
     model.summary()
     training_generator=IEMR()
     valid_generator=IEMR_valid()
     
     earlyStopping = EarlyStopping(monitor='loss', patience=20, verbose=0, mode='min')
     mcp_save = ModelCheckpoint('./0428_data??0000.hdf5', save_best_only=True, monitor='loss', mode='min')
     reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
     history=keras.callbacks.History()
     optimizer = keras.optimizers.Adam(lr=0.01)
     model.compile(optimizer=optimizer,loss= 'categorical_crossentropy',metrics=['accuracy'])

     history1=keras.callbacks.History()
     model.fit_generator(training_generator,steps_per_epoch=1000,epochs=100,validation_data=valid_generator,validation_steps=10,callbacks=[earlyStopping, mcp_save, reduce_lr_loss,history])
     result=model.predict_generator(valid_generator,steps=1000, max_queue_size=10, workers=1,use_multiprocessing=False, verbose=0,callbacks=[history1])
     print(result)
  
     print(history.history.keys())
   
     

   
     