import pandas as pd
file_name='./emotion.xlsx'
dfs = pd.read_excel(file_name, sheet_name='Sheet1')
print(dfs['Subject'][0])

mapping={
            'happiness':0,
            'surprise':1,
            'disgust':2,
            'others':3,
            'repression':4,
            'sadness':5,
            'fear':6,
        }
import os
import numpy as np
import json
w= open('./train_emotion_casme0.8.json','w')
 
path='../../../attributes/'
subs=os.listdir(path)
data=[]
count=0
for txt in subs:
    count=count+1
    sub=txt[3:5]
    track=txt[6:-4]
    #print(sub+'\n')
    #print(track+'\n')
    for index,i in enumerate(dfs['Subject']):
      if int(sub)==int(i) and track==dfs['Filename'][index]:
        emotion=dfs['Estimated Emotion'][index]
      else:
        continue
    print(emotion)
    if count%5==0:
      continue
    f=open(path+txt,'r')
    lines=f.readlines()
    arr=[]
    for line in lines:
        #print(line)
        words=line.split()
        seq=[]
        for word in words:
            #print(word)
            if word[0]=='s':
              continue
            seq.append(word)
        arr.append(seq)
    arrr=np.array(arr)
    #print(np.shape(arrr))
    if np.shape(arrr)[0]<50:
      print('found')
      continue
    
    anno={
            
            'emotion':mapping[emotion],
            'intensity':arr
         }
    #print(anno)
    data.append(anno)
print(len(data))
json.dump(data,w)  
  
  
  
  
  
  
#Subject  Filename  Unnamed: 2  OnsetFrame ApexFrame  OffsetFrame  Unnamed: 6 Action Units Estimated Emotion
