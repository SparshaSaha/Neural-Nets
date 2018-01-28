import numpy as np
import matplotlib.pyplot as mplot

def get_data(x):
    datafile=open(x,"r")
    datalist=[]
    for line in datafile:
        z=line.split(",")
        x=[int (i) for i in z]
        datalist.append(x)
    f=np.matrix(datalist)
    return datalist

def signum(data):
    if data<0:
        return 0
    else:
        return 1

def get_weight_dim(x):
    return (len(x[0]))


def multiply(x,w):
    return np.matmul(x,w).tolist()



 
x=get_data("x_data.txt")
for i in x:
    i.insert(0,1)
y=get_data("y_data.txt")
dim=get_weight_dim(x)
weightvec=[[1] for i in range(dim)]
print(multiply(x[1],weightvec))


