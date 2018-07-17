import numpy as np
import matplotlib.pyplot as plt 

data=np.loadtxt("D:/Users/Tristan_local/OneDrive/myData/Machine Learning 2 2018/3A/data/data.txt",delimiter=',')
x=data[:,0]
y=data[:,1]
#beta_0=np.mean

xMean=np.mean(x)
yMean=np.mean(y)

def returnMean(mean):
	return yMean

#pPearson Product Moments
PPM=np.corrcoef(x,y)

#Affine Prognose auf Basis von X (teachers salary)
xAxis=np.linspace(min(x),max(x), 50)
beta1_x=np.cov(x,y)[0,1]/(np.var(x))
beta0_x=np.mean(y)-beta1_x*np.mean(x)

def y_X(m):
	return beta1_x*m+beta0_x
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(xAxis,y_X(xAxis),label='y(X)')
ax1.plot(x,y,color='green', marker='o', linestyle='',label='data')
plt.legend(loc='lower left');
plt.xlabel('X')        
plt.ylabel('Y')
plt.show()


#Affine Prognose auf Basis von Y (SAT score)
yAxis=np.linspace(min(y),max(y), 50)
beta1_y=np.cov(x,y)[0,1]/(np.var(y))
beta0_y=np.mean(x)-beta1_y*np.mean(y)

def y_Y(m):
	return beta1_y*m+beta0_y
fig = plt.figure()
ax2 = fig.add_subplot(111)
ax2.plot(yAxis,y_Y(yAxis),label='y(Y)')
ax2.plot(y,x,color='green', marker='o', linestyle='',label='data')
plt.legend(loc='lower left');
plt.xlabel('Y')        
plt.ylabel('X')
plt.show()


