
import json
import numpy as np
import os 
f= open('./train_intensity.json','r')


train_dict=json.load(f)
train_intensity=[]
for j in range(100):
  for i,dicti in enumerate(train_dict):
    #print(len(train_dict[i]['intensity']))
      temp1=[]
      for j in train_dict[i]['intensity']:
        temp2=[]
        for k in j:
          temp2.append(int(k))
        temp1.append(temp2)
      train_intensity.append(temp1)
      print(len(train_intensity))


