import pandas as pd
import numpy as np
from PIL import Image
import json
import random
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from keras.utils import Sequence
import tensorflow as tf
f=open('test_intensity.json','r')
train_dict=json.load(f)
train_intensity=[]

for i,dicti in enumerate(train_dict):
    #print(len(train_dict[i]['intensity']))
    temp1=[]
    for j in train_dict[i]['intensity']:
      #print(j)
      temp2=[]
      for k in j:
        temp2.append(int(k))
      temp1.append(temp2)
    for j in range(10000):
      train_intensity.append(temp1)
print(len(train_intensity))
class IEMR_valid(Sequence):
    def __init__(self):

        self.to_tensor = transforms.ToTensor()
        self.intensity_arr = train_intensity
        print(len(self.intensity_arr))
        print(len(self.intensity_arr[1]))
        print(len(self.intensity_arr[1][0]))
        self.data_len = 4845
        self.temp=[]
        '''self.switch = {
            '1':self.__scaling__(seq=self.temp,param=0.25),#Scaling0.25
            '2':self.__scaling__(seq=self.temp,param=0.5),#Scaling0.5
            '3':self.__scaling__(seq=self.temp,param=1.25),#Scaling1.25
            '4':self.__scaling__(seq=self.temp,param=1.7),#Scaling1.7
            '5':self.__GausianNoise__(seq=self.temp,param=0.1),#GaussianN0.1
            '6':self.__GausianNoise__(seq=self.temp,param=0.25),#GaussianN0.25
            '7':self.__GausianNoise__(seq=self.temp,param=0.5),#GaussianN0.5
            '8':self.__GausianNoise__(seq=self.temp,param=0.75),#GaussianN0.75
            '9':self.__GausianNoise__(seq=self.temp,param=0.9),#GaussianN0.9
            '10':self.__ZeroFill__(seq=self.temp),#ZeroFill
            '11':self.__Noned__(seq=self.temp),#None
            }'''
        self.switch = {
            '1':self.__scaling__,#Scaling0.25
            '2':self.__scaling__,#Scaling0.5
            '3':self.__scaling__,#Scaling1.25
            '4':self.__scaling__,#Scaling1.7
            '5':self.__GausianNoise__,#GaussianN0.1
            '6':self.__GausianNoise__,#GaussianN0.25
            '7':self.__GausianNoise__,#GaussianN0.5
            '8':self.__GausianNoise__,#GaussianN0.75
            '9':self.__GausianNoise__,#GaussianN0.9
            '10':self.__ZeroFill__,#ZeroFill
            '11':self.__Noned__,#None
            }
        self.param = {
            '1':0.25,#Scaling0.25
            '2':0.5,#Scaling0.5
            '3':1.25,#Scaling1.25
            '4':1.7,#Scaling1.7
            '5':0.1,#GaussianN0.1
            '6':0.25,#GaussianN0.25
            '7':0.5,#GaussianN0.5
            '8':0.75,#GaussianN0.75
            '9':0.9,#GaussianN0.9
            '10':None,#ZeroFill
            '11':None,#None
            }
        self.n=0
        self.max=self.__len__()       
    def __choose_transformation__(self,index,array):
        signal= random.randint(1,10)
        array=np.asarray(array)
        array=array.T.tolist()
        out=[]
        for i,seq in enumerate(array):
          #print(i)
          if((i>=index)and(i<index+4000)):
            temp=seq
            seq= self.switch["%s"%signal](temp,self.param["%s"%signal])
            out.append(seq)
          else:
            out.append(seq) 
        out=np.asarray(out)
        out=out.T.tolist()

        return out,signal
             
    def __getitem__(self, index):

        single_intensity_sequence = self.intensity_arr[index]
 
        total=0

        
        start= random.randint(0,self.data_len-1500)
        #start=0
        end=start+1500
        index= random.randint(1,500)
        #index=1
        #print(index) 
        intensity_arr=[]
        for i in single_intensity_sequence:
          j=i[start:end]
          j=[k * 5 for k in j]
          intensity_arr.append(j)
        transformed_arr,signal=self.__choose_transformation__(index,intensity_arr)

        
         
       
        intensity_arr=np.asarray(transformed_arr)
        intensity_arr=np.expand_dims(intensity_arr, axis=0)
        intensity_arr=np.expand_dims(intensity_arr, axis=3)
      
        index_label=np.zeros(500)
        index_label[index-1]=1
        signal_label=np.zeros(10)
        signal_label[signal-1]=1

        index_label=index_label.astype('float32')
        signal_label=signal_label.astype('float32')
        index_label=tf.convert_to_tensor(index_label)
        signal_label=tf.convert_to_tensor(signal_label)
        index_label=tf.expand_dims(index_label,axis=0)
        signal_label=tf.expand_dims(signal_label,axis=0)

        return intensity_arr,[index_label, signal_label]
    
    def __next__(self):
        if self.n >= self.max:
           self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        #print(self.n)
        return result
    def __len__(self):
        return len(self.intensity_arr)
    def __scaling__(self,seq,param):
     
      #print(param)
      seq= [i * param for i in seq]
      #seq=['test']
      return seq
    def __GausianNoise__(self,seq,param):
      #print('G')
      #print(param)
      seq=np.asarray(seq)
      seq = seq+np.random.normal(0, param, seq.shape)
      #seq = np.clip(seq, 0, 255)
      return seq
    def __ZeroFill__(self,seq,param):
      #print('Z')
      seq=np.asarray(seq)
      seq=np.zeros(seq.shape)
      return seq
    def __Noned__(self,seq,param):
      #print('N')
      seq=np.asarray(seq)
      temp=[]
      for i in  range(0,len(seq)):
        temp.append(None)
      seq=np.asarray(temp)        
      return seq



if __name__=="__main__":
    dataset=IEMR()
    dataset.__getitem__(0)
    training_generator = IEMR()
    #validation_generator = DataGenerator(val_df, val_idx, **params)
    print(training_generator) 
