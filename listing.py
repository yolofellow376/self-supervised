
import json

f= open('./test_intensity.json','w')

import numpy as np
import os 
root='./ActionUnit_Labels/'
subs=os.listdir(root)
data=[]
i=0
for sub in subs:
    i=i+1
    if i%5!=0:
      continue
    print(sub)
    aus=os.listdir(root+sub)
    arr=[]
    for au in aus:
        print(au[6:-4])
        f2=open(root+sub+'/'+au)
        lines=f2.readlines()
        #print(lines)
        seq=[]
        for line in lines:
            ch=line.split(',')
            seq.append(int(ch[1]))
        sequ=np.array(seq)
        arr.append(seq)
    arrr=np.array(arr)
    print(np.shape(arrr))
    anno={
            'sub':sub,
            'intensity':arr
         }
    #print(anno)
    data.append(anno)
        
json.dump(data,f)
