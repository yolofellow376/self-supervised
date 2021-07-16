def scaling(seq,param):
  print(seq)
  print(param)
def GausianNoise(param):
  print('G')
  print(param)
switch = {
    "1":scaling,#Scaling0.25
    "2":scaling,#Scaling0.5
    "3":scaling,#Scaling1.25
    "4":scaling,#Scaling1.7
    "5":GausianNoise,#GaussianN0.1
    "6":GausianNoise,#GaussianN0.25
    "7":GausianNoise,#GaussianN0.5
    "8":GausianNoise,#GaussianN0.75
    "9":GausianNoise,#GaussianN0.9
    "10":lambda x:x**x,#ZeroFill
    "11":lambda x:x**x,#None

}
l=[0,0,0,0,0]
try:
    switch["%s"%1](l,0.25)
except KeyError as e:
    pass