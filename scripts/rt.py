import random
import numpy as np

array=np.zeros((10,1))
print(array.shape)
for i in range(10000):
  key=random.randint(0,9)
  array[key][0]=array[key][0]+1
  
print(array)
  
