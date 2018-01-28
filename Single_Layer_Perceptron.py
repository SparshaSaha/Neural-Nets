import numpy as np
import matplotlib.pyplot as plt
import random as rand

def get_data(x):
    datafile=open(x,"r")
    datalist=[]
    for line in datafile:
        z=line.split(",")
        x=[float(i) for i in z]
        datalist.append(x)
    f=np.matrix(datalist)
    return datalist

def signum(data):
    if data<0:
        return -1
    else:
        return 1

def get_weight_dim(x):
    return (len(x[0]))


def multiply(x,w):
    return np.matmul(x,w)

def training(iters,learning_rate,x,y,weight):
    curr_row=0
    for i in range(iters):
        temp=multiply(x[curr_row],weight)
        temp=signum(temp[0,0])
        
        if temp!=y[curr_row,0]:
            weight=weight+learning_rate*(y[curr_row,0]-temp)*(x[curr_row].transpose())
            

        curr_row=curr_row+1
        if curr_row==len(x):
            curr_row=0
        
        

        
    return weight
    
    
def calc_error(x,y,weight):
   f=multiply(x,weight)
   f=f.tolist()
   print(f)
    

 
x=get_data("x_data.txt")
for i in x:
    i.insert(0,1)
y=get_data("y_data.txt")
plt.plot([i[1] for i in x],[i[2] for i in x],"ro")

dim=get_weight_dim(x)
weightvec=[[rand.uniform(-1,1)] for i in range(dim)]
weightvec=training(1000,0.01,np.matrix(x),np.matrix(y),np.matrix(weightvec))
calc_error(np.matrix(x),np.matrix(y),weightvec)
#plt.show()

