from myloader import IEMR
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense,AvgPool2D,DepthwiseConv2D,SeparableConv2D
from tensorflow.keras import Model
import numpy as np
from keras import backend as K
import argparse
from keras.models import Sequential
gpus= tf.config.list_physical_devices('GPU')
print(gpus)
tf.config.experimental.set_memory_growth(gpus[0], True)

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
  index_label=index_label.astype('float32')
  signal_label=signal_label.astype('float32')
  index_label=tf.convert_to_tensor(index_label)
  signal_label=tf.convert_to_tensor(signal_label)

       
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
 
    
     model = Model(inputs=input, outputs=output)
     model.summary()
    
     training_generator=IEMR()
     
     optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
     model.compile(optimizer=optimizer)
     count=0
     for epoch in range(0,200):
       for data_batch, labels in training_generator:
         
         loss_value, grads = grad(model, data_batch, labels)
         print("\nK.categorical_crossentropy:",loss_value)
         #print("\ntf.GradientTape.gradient(loss,model.trainable_variables):\n" ,grads)
         optimizer.apply_gradients(zip(grads, model.trainable_variables))
         count=count+1
     print(count)
       
