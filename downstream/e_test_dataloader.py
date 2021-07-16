import pandas as pd
import numpy as np
from PIL import Image
import json
import random
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from keras.utils import Sequence
import tensorflow as tf
f=open('test_emotion_casme0.2.json','r')
train_dict=json.load(f)
train_intensity=[]
emotion_label=[]
for i,dicti in enumerate(train_dict):
    #print(len(train_dict[i]['intensity']))
    temp1=[]
    for j in train_dict[i]['intensity']:
      temp2=[]
      for k in j:
        temp2.append(float(k))
      temp1.append(temp2)
    temp1=np.asarray(temp1)
    temp1=np.transpose(temp1)
    
    for j in range(100):
      train_intensity.append(temp1)
      emotion_label.append(train_dict[i]['emotion'])

class IEMR_Emotion_valid(Sequence):
    def __init__(self):

        self.to_tensor = transforms.ToTensor()
        self.intensity_array = train_intensity
        self.emotion_array = emotion_label
        print(len(self.intensity_array))
        print(len(self.intensity_array[0]))
        print(len(self.intensity_array[0][0]))
        self.data_len = 4845
        self.temp=[]

             
    def __getitem__(self, index):

        single_intensity_sequence = self.intensity_array[index]
        emotion=self.emotion_array[index]
        #print(len(single_intensity_sequence[0]))
        total=0
        
        
        start= random.randint(0,len(single_intensity_sequence[0])-50)
        #start=0
        end=start+50
        
        intensity_arr=[]
        for i in single_intensity_sequence:
          j=i[start:end]
          intensity_arr.append(j)
        #print("array")
        #print(intensity_arr)
        #print(intensity_arr.)
        

        
         
       
        intensity_arr=np.asarray(intensity_arr)
        #print(intensity_arr.shape)
        intensity_arr=np.expand_dims(intensity_arr, axis=0)
        intensity_arr=np.expand_dims(intensity_arr, axis=3)
        emotion_label=np.zeros(7)
        emotion_label[emotion]=1
        emotion_label=emotion_label.astype('float32')
        emotion_label=tf.convert_to_tensor(emotion_label)
        emotion_label=tf.expand_dims(emotion_label,axis=0)
        #print("emotions")
        #print(emotion_label)
        return intensity_arr,emotion_label
    
    def __next__(self):
        if self.n >= self.max:
           self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        #print(self.n)
        return result
    def __len__(self):
        return len(self.intensity_array)





if __name__=="__main__":
    dataset=IEMR_Emotion_valid()
    a=dataset.__getitem__(10)
    print(a)
    training_generator = IEMR_Emotion_valid()
    #validation_generator = DataGenerator(val_df, val_idx, **params)
    print(training_generator) 
