from myloader import IEMR
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense,AvgPool2D,DepthwiseConv2D,SeparableConv2D
from tensorflow.keras import Model
import numpy as np
from keras import backend as K
import argparse
from keras.models import Sequential
import keras
###############################

#################################

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
    tape.watch(model.variables)

    grad=tape.gradient(loss_value, model.variables)
 
  return loss_value, grad

def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)
  
  index_predict=y_[0]
  signal_predict=y_[1]
  #print("index_predict:",index_predict[0])
  index_label=np.zeros(100)
  index_label[y[0][0]]=1
  #print("index_label:",index_label)
  #signal_predict=np.squeeze(signal_predict)
  #print("signal_predict:",signal_predict[0])
  signal_label=np.zeros(12)
  signal_label[y[0][1]]=1
  #print("signal_label:",signal_label)
  index_label=index_label.astype('float64')
  signal_label=signal_label.astype('float64')
  index_label=tf.convert_to_tensor(index_label)
  signal_label=tf.convert_to_tensor(signal_label)
  index_label=K.constant(index_label)
  signal_label=K.constant(signal_label)
       
  index_weight=1
  signal_weight=1
       
  g1=K.categorical_crossentropy(target=index_label,output=index_predict[0])
  g2=K.categorical_crossentropy(target=signal_label,output=signal_predict[0])
  #print(g1)
  #print(g2)     

  #for batch>1
  #log_loss=index_weight*np.sum(ce1)/ce1.shape[0]+signal_weight*np.sum(ce2)/ce2.shape[0]
  log_loss=index_weight*g1+signal_weight*g2
  log_loss=tf.convert_to_tensor(log_loss) 
  #print(log_loss)
  return log_loss

if __name__=="__main__":
     parser = argparse.ArgumentParser(description='Process some integers.')
     parser.add_argument('--downstream', default=None)
     args = parser.parse_args()
     model = keras.models.load_model("my_model.h5")
     model.summary()
     if args.downstream=='RF':
       print('random forest')
       model = keras.models.load_model("my_model")
       for layer in model.layers:
          layer.trainable = False
       model.add(Dense(output_dim = 256, activation = 'relu'))
       model.add(Dense(output_dim = 128, activation = 'relu'))
       model.add(Dense(output_dim = 1, activation = 'relu'))
     elif args.downstream=='3FC':
       print('3FC')
       model = keras.models.load_model("my_model")
       for layer in model.layers:
          layer.trainable = False
       model.add(Dense(output_dim = 256, activation = 'relu'))
       model.add(Dense(output_dim = 128, activation = 'relu'))
       model.add(Dense(output_dim = 1, activation = 'relu'))
     model.summary()
 
     