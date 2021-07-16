import os
import numpy as np
import json
w= open('./train_intensity_casme.json','w')

path='../attributes/'
subs=os.listdir(path)
data=[]
i=0
for txt in subs:
    i=i+1
    if i%10==0:
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
    print(np.shape(arrr))
    if np.shape(arrr)[0]<50:
      print('found')
      continue
    anno={
            
            'intensity':arr
         }
    #print(anno)
    data.append(anno)
print(len(data))
json.dump(data,w)