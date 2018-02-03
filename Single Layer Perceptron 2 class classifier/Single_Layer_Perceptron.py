#Code by Sparsha
import numpy as np
import matplotlib.pyplot as plt
import random as rand


def get_data(x):     #function to read data from file
    datafile=open(x,"r")
    datalist=[]
    for line in datafile:
        z=line.split(",")
        x=[float(i) for i in z]
        datalist.append(x)
    f=np.matrix(datalist)
    return datalist



def signum(data):    #activation function signum
    if data < 0:
        return -1
    else:
        return 1




def get_weight_dim(x):  #get dimensions of x data to calculate dimensions of weight matrix
    return (len(x[0]))




def multiply(x,w):      #multiply two matrices
    return np.matmul(x,w)




def training(iters,learning_rate,x,y,weight):  #function to train model
    curr_row=0
    for i in range(iters):
        temp=multiply(x[curr_row],weight)
        temp=signum(temp[0,0])
        
        if temp!=y[curr_row,0]:
            weight=weight+learning_rate*(y[curr_row,0]-temp)*(x[curr_row].transpose())
            

        curr_row=curr_row+1
        if curr_row==len(x):
            curr_row=0
        
        
        z=calc_error(x,y,weight)
        print("Iteration : "+str(i)+"  error : "+str(z))
        if z==0.0:
            break
        
    return weight
    
def calc_error(x,y,weight):  #function to calculate error
    f=multiply(x,weight)
    f=f.tolist()
    f=[[signum(i[0]) ] for i in f]
    c=y-f
    error=0
    for i in range(len(c)):
        if c[i][0]!=0.0:
            error=error+1
    return (error/len(x) *100)



#function calling
x=get_data("x_data.txt") #Name of the file containing the x_data set where the features are separated by commas
for i in x:
    i.insert(0,1)
y=get_data("y_data.txt") #Name of the file containing the y_data set
plt.plot([i[1] for i in x],[i[2] for i in x],"bx")

dim=get_weight_dim(x)
weightvec=[[rand.uniform(-1,1)] for i in range(dim)] #Random initialization of weights at the begining between -1 and +1
weightvec=training(1000,0.01,np.matrix(x),np.matrix(y),np.matrix(weightvec))
print('\n')
print("Weights are: ")
print(weightvec) #Final weights
print('\n')
plt.show()

