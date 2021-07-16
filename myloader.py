import pandas as pd
import numpy as np
from PIL import Image
import json
import random
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from keras.utils import Sequence
import tensorflow as tf
f=open('train_intensity.json','r')
train_dict=json.load(f)
train_intensity=[]
for i,dicti in enumerate(train_dict):
    #print(len(train_dict[i]['intensity']))
    temp1=[]
    for j in train_dict[i]['intensity']:
      temp2=[]
      for k in j:
        temp2.append(int(k))
      temp1.append(temp2)
    for j in range(100):
      train_intensity.append(temp1)


class IEMR(Dataset):
    def __init__(self):

        self.to_tensor = transforms.ToTensor()
        self.intensity_arr = train_intensity
        #print(len(self.intensity_arr))
        #print(len(self.intensity_arr[1]))
        #print(len(self.intensity_arr[1][0]))
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
        #signal=11
        array=np.asarray(array)
        array=array.T.tolist()
        #print('checkpoint')
        #print(index)
        out=[]
        for i,seq in enumerate(array):
          #print(i)
          if((i>=index)and(i<index+25)):
            self.temp=seq
            #print(self.temp)
            seq= self.switch["%s"%signal](self.temp,self.param["%s"%signal])
            #print(seq)
            out.append(seq)
          else:
            out.append(seq) 
        out=np.asarray(out)
        out=out.T.tolist()
        #print('checkpoint')
        return out,signal
    def __getitem__(self, index):
        # Get image name from the pandas df
        single_intensity_sequence = self.intensity_arr[index]
        #print((single_intensity_sequence[0]))
        # Open image
        start= random.randint(0,self.data_len-100)
        end=start+100
        index= random.randint(1,80)
        #print(index) 
        intensity_arr=[]
        for i in single_intensity_sequence:
          intensity_arr.append(i[start:end])
        #print(index)
        #intensity_arr=self.transform(single_intensity_sequence[index:index+10][:])
        transformed_arr,signal_label=self.__choose_transformation__(index,intensity_arr)
             
        
        index_label=index 
        #signal_label=transform_to_label[str(self.transform)]
        #print(transformed_arr)
        #print(index_label)
        #print(signal_label)
        intensity_arr=np.asarray(intensity_arr)
        index_label=np.asarray(index_label)
        signal_label=np.asarray(signal_label)
        labels=[]
        labels.append(index_label)
        labels.append(signal_label)
        labels=np.asarray(labels)
        
        intensity_arr=np.expand_dims(intensity_arr, axis=0)


        intensity_arr=tf.convert_to_tensor(intensity_arr)
        labels=tf.convert_to_tensor(labels)
        
        
        
        return intensity_arr, labels
    
    def __next__(self):
        if self.n >= self.max:
           self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        print(self.n)
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
'''    def _generate_X(self, list_IDs_temp):
        """Generate every batch of images
                 :param list_IDs_temp: batch data index list
                 :return: a batch of images
        """
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        for i, ID in enumerate(list_IDs_temp):
                         # Store a batch
            X[i,] = self._load_image(self.df.iloc[ID].images)
        return X
 
 
   def _generate_y(self, list_IDs_temp):
        """Generate labels for each batch
                 :param list_IDs_temp: batch data index list
                 :return: a batch label
        """
        y = np.empty((self.batch_size, self.n_classes), dtype=int)
        for i, ID in enumerate(list_IDs_temp):
            y[i,] = self._labels_encode(self.df.iloc[ID].labels, config.LABELS)
        return y
        
    def on_epoch_end(self):
        """Update the index after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def _load_image(self, image_path):
        """cv2 read image
        """
        # img = cv2.imread(image_path)
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        w, h, _ = img.shape
        if w>h:
            img = np.rot90(img)
        img = cv2.resize(img, (472, 256))
        return img
        
    def _labels_encode(self, s, keys):
        """ tag one-hot encoding conversion
        """
        cs = s.split('_')
        y = np.zeros(13)
        for i in range(len(cs)):
            for j in range(len(keys)):
                for c in cs:
                    if c == keys[j]:
                        y[j] = 1
        return y'''



if __name__=="__main__":
    dataset=IEMR()
    dataset.__getitem__(0)
    training_generator = IEMR()
    #validation_generator = DataGenerator(val_df, val_idx, **params)
    print(training_generator) 
