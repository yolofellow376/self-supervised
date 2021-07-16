from e_train_dataloader import IEMR_Emotion
from e_test_dataloader import IEMR_Emotion_valid
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
#x = Dropout(0.5)(x)

x = Conv2D(filters=32, kernel_size=(1,17), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = AvgPool2D(pool_size=2, strides=2,padding='same')(x)
x = Dropout(0.5)(x)

x = Conv2D(filters=64, kernel_size=(1,17),padding='same',activation='relu')(x)
x = BatchNormalization()(x)
#x = AvgPool2D(pool_size=2, strides=2,padding='same')(x)
x = Dropout(0.5)(x)

x = Conv2D(filters=64, kernel_size=(1,17),padding='same',activation='relu')(x)
x = BatchNormalization()(x)
#x = AvgPool2D(pool_size=2, strides=2,padding='same')(x)
x = Dropout(0.5)(x)



x = DepthwiseConv2D(kernel_size=(1,17),depth_multiplier=1,padding='same')(x)
x = BatchNormalization()(x)
#x = AvgPool2D(pool_size=2, strides=2,padding='same')(x)
#x = Dropout(0.5)(x)

x = SeparableConv2D(filters=64, kernel_size=1,depth_multiplier=1)(x)
x = BatchNormalization()(x)
#x = AvgPool2D(pool_size=2, strides=2,padding='same')(x)
#x = Dropout(0.5)(x)


out = Flatten()(x)
#out = Dropout(0.5)(out)
window_output= Dense(units=15, activation='softmax')(out)
#window_output=Dropout(0.5)(window_output)
#window_output = BatchNormalization()(window_output)

signal_output= Dense(units=12, activation='softmax')(out)
#signal_output = BatchNormalization()(signal_output)
#signal_output=Dropout(0.5)(signal_output)
#output = Dense(units=12, activation='softmax')(x)

output=[window_output,signal_output]
#################################

if __name__=="__main__":
     parser = argparse.ArgumentParser(description='Process some integers.')
     parser.add_argument('--downstream', default=None)
     args = parser.parse_args()

     model = Model(inputs=input, outputs=[window_output,signal_output])   
     model.load_weights("./0611two_task.hdf5")
     model.summary()
     if args.downstream=='3FC':
       print('3FC')
       new_model=Sequential()
       for layer in model.layers[:-2]:
          layer.trainable = False
          print(str(layer)[0:26])#<keras.layers.core.Dropout object at 0x7fbf9c4d9250>
          if(str(layer)[0:26]=="<keras.layers.core.Dropout"):
            print("!!!!")
            continue
          new_model.add(layer)
       model=new_model
       #model.add(BatchNormalization())
       #model.add(Dense(units = 1024, activation = 'relu'))
       #model.add(Dense(units = 512, activation = 'relu'))
       model.add(Dense(units = 256, activation = 'relu'))
       model.add(Dropout(0.4))
       model.add(Dense(units = 128, activation = 'relu'))
       model.add(Dropout(0.4))
       model.add(Dense(units = 7, activation = 'softmax'))
       for layer in model.layers:
          layer.trainable = False
       #model.add(Dropout(0.1))   
     model.summary()
     
     
     
     earlyStopping = EarlyStopping(monitor='loss', patience=20, verbose=0, mode='min')
     mcp_save = ModelCheckpoint('./TEMP.hdf5', save_best_only=True, monitor='loss', mode='min')
     reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
     history=keras.callbacks.History()
     log_dir = "logs275/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
     tensorboard = TensorBoard(log_dir="logs")
     optimizer = keras.optimizers.SGD(lr=0.01)
     model.compile(optimizer=optimizer,loss= 'categorical_crossentropy',metrics=['accuracy'])
     training_generator=IEMR_Emotion()
     valid_generator=IEMR_Emotion_valid()
     model.fit_generator(training_generator,steps_per_epoch=1000,validation_steps=1000,validation_data=training_generator,epochs=300,callbacks=[earlyStopping, mcp_save, reduce_lr_loss,history,tensorboard], shuffle=True)
     
   
     #,steps_per_epoch=100,validation_steps=100
