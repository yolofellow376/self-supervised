from train_dataloader import IEMR
from test_dataloader import IEMR_valid
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense,AvgPool2D,DepthwiseConv2D,SeparableConv2D,BatchNormalization,Dropout
from keras import Model
import numpy as np
from keras import backend as K
import argparse
from keras.models import Sequential
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping,ModelCheckpoint, ReduceLROnPlateau
import keras
import datetime
from keras.callbacks import TensorBoard
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
input = Input(shape=(17,50,1))
x = Conv2D(filters=16, kernel_size=(1,17), padding='valid', activation='relu')(input)
x = BatchNormalization()(x)
x = AvgPool2D(pool_size=2, strides=2, padding='valid')(x)
x = Dropout(0.1)(x)

x = Conv2D(filters=32, kernel_size=(1,17), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = AvgPool2D(pool_size=2, strides=2,padding='same')(x)
x = Dropout(0.1)(x)

x = Conv2D(filters=64, kernel_size=(1,17),padding='same',activation='relu')(x)
x = BatchNormalization()(x)
#x = AvgPool2D(pool_size=2, strides=2,padding='same')(x)
x = Dropout(0.1)(x)

x = Conv2D(filters=64, kernel_size=(1,17),padding='same',activation='relu')(x)
x = BatchNormalization()(x)
#x = AvgPool2D(pool_size=2, strides=2,padding='same')(x)
x = Dropout(0.1)(x)



x = DepthwiseConv2D(kernel_size=(1,17),depth_multiplier=1,padding='same')(x)
x = BatchNormalization()(x)
#x = AvgPool2D(pool_size=2, strides=2,padding='same')(x)
x = Dropout(0.1)(x)

x = SeparableConv2D(filters=64, kernel_size=1,depth_multiplier=1)(x)
x = BatchNormalization()(x)
#x = AvgPool2D(pool_size=2, strides=2,padding='same')(x)
x = Dropout(0.1)(x)


out = Flatten()(x)
#out = Dropout(0.5)(out)
window_output= Dense(units=15, activation='softmax')(out)
#window_output=Dropout(0.5)(window_output)
#window_output = BatchNormalization()(window_output)

signal_output= Dense(units=12, activation='softmax')(out)
#signal_output = BatchNormalization()(signal_output)
#signal_output=Dropout(0.1)(signal_output)
#output = Dense(units=12, activation='softmax')(x)

output=[window_output,signal_output]
#################################

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

    
     model = Model(inputs=input, outputs=[window_output,signal_output])   
     model.summary()
     training_generator=IEMR()
     valid_generator=IEMR_valid()
     
     earlyStopping = EarlyStopping(monitor='loss', patience=20, verbose=0, mode='min')
     mcp_save = ModelCheckpoint('./0611_twotask_1.hdf5', save_best_only=True, monitor='loss', mode='min')
     reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, verbose=1, epsilon=1e-4, mode='min')
     history=keras.callbacks.History()
     optimizer = keras.optimizers.Adam(lr=0.0001)
     model.compile(optimizer=optimizer,loss= 'categorical_crossentropy',metrics=['accuracy'])
     log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
     tensorboard = TensorBoard(log_dir="logs")
     history1=keras.callbacks.History()
     model.fit_generator(training_generator,steps_per_epoch=1000,epochs=300,validation_data=valid_generator,validation_steps=1000,callbacks=[earlyStopping, mcp_save, reduce_lr_loss,history,tensorboard], shuffle=True)
     #result=model.predict_generator(valid_generator,steps=1000, max_queue_size=10, workers=1,use_multiprocessing=False, verbose=0,callbacks=[history1])
     #print(result)
  
     #print(history.history.keys())
   
     

   
     