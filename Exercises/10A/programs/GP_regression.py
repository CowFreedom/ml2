import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt 
import random


		

class GP:
	def __init__(self,sigmaF,choice):
		self.sigmaF=sigmaF
		self.choice=choice

		
	def neural_net_kernel(self,x,y):
		Sigma=8*np.eye(len(x))
		nominator=2*x.T.dot(Sigma).dot(y)
		denominator=np.sqrt((1+2*(x.T.dot(Sigma).dot(x)))*(1+2*(y.T.dot(Sigma).dot(y))))
		return (2/np.pi)*np.arcsin((nominator)/(denominator))
	
	def rbf_kernel(self,x,y):
		l=self.param[0]
		return self.sigmaF**2*np.exp(-(0.5/(l**2))*np.linalg.norm(x-y)**2)
		
	def calculate_kernel(self,x,y):
		cov=np.zeros(shape=(len(x),len(y)))	
		for i in range(len(x)):
			for j in range(len(y)):
				cov[i,j]=self.kernel(x[i],y[j])
		return cov		
	
	def calculate_cov(self):
		kern1=self.calculate_kernel(self.X,self.X)
		kern2=self.calculate_kernel(self.Xp,self.X)	
		kern3=self.calculate_kernel(self.Xp,self.Xp)	
		kern4=self.calculate_kernel(self.X,self.Xp)	
		return kern3-kern2.dot(inv((kern1+self.sigmaF*np.eye(len(kern1))))).dot(kern4)
		#return kern1
	
	def calculate_mean(self):
		kern1=self.calculate_kernel(self.X,self.X)
		kern2=self.calculate_kernel(self.Xp,self.X)
		#print("SHAPE:",inv((kern1+self.sigmaF*np.eye(len(kern1)))).shape)
		return kern2.dot(inv((kern1+self.sigmaF*np.eye(len(kern1))))).dot(self.t)

	def basis_function(self,x_0,mu):
		k=1 #affects locality of gaussians
		return np.exp(-0.5*k*(x_0-mu)*(x_0-mu))

	def populate_design_matrix(self,x,X,n): #Defining the design matrix
		i_p=(self.rangeStart-self.rangeEnd)/(self.m)
		for i in range(n):
			for j in range(1,self.m):
				X[i,j]=self.basis_function(x[i],self.rangeStart+j*i_p)
					
			
	def eval(self,x,t,xp,param,m):
		self.t=t
		self.X=np.ones(shape=(len(x),m)) #design matrix
		self.Xp=np.ones(shape=(len(xp),m)) #design matrix
		self.m=m#number of basis function in design matrix
		self.rangeStart=min(x)
		self.rangeEnd=max(x)
		self.param=param
		if self.choice==0:
			self.kernel=self.rbf_kernel
		else:
			self.kernel=self.neural_net_kernel
		
		self.populate_design_matrix(x,self.X,len(x))	
		self.populate_design_matrix(xp,self.Xp,len(xp))
		#cov=self.calculate_cov()	
		mean=self.calculate_mean()		
		cov=self.calculate_cov()
		
		
		return np.random.multivariate_normal(mean,cov)
		
		
		
		
def example1(n,k):
	#generate function values
	sigmaF=0.1 #standard deviation of an observed value t, s.d. t~N(y(x),\sigma)
	x=np.sort(np.random.uniform(0,1,n))
	f=lambda x:np.sin(2*np.pi*x)
	t=np.array([f(k)+np.random.normal(0,sigmaF) for k in x])
	xp=np.sort(np.random.uniform(0,1,k))#x positions to be evaluated by the GP
	m=10#number of basis functions

	gp=GP(sigmaF,0)
	
	
	#Plotting
	rep=3
	plt.plot(x,t,color=(random.random(),random.random(),random.random()), marker='o', linestyle='--')	
	plt.legend(["f(x)+e"],loc='lower left')	
	#plt.title("Neural Network (depth: "+ str(depth)+" h: "+str(h)+") vs. Gaussian Process")
	for i in range(rep):
		plt.plot(xp,gp.eval(x,t,xp,[0.31],m),color=(random.random(),random.random(),random.random()), marker='', linestyle='-')	
	plt.legend(["samples"],loc='lower left')

	plt.xlabel('x')        
	plt.ylabel('y')

	plt.show()
	
def example2(n,k):
	#generate function values
	sigmaF=0.1 #standard deviation of an observed value t, s.d. t~N(y(x),\sigma)
	x=np.sort(np.random.uniform(0,1,n))
	f=lambda x:np.sin(2*np.pi*x)
	t=np.array([f(k)+np.random.normal(0,sigmaF) for k in x])
	xp=np.sort(np.random.uniform(0,1,k))#x positions to be evaluated by the GP
	m=10 #number of basis functions

	gp=GP(sigmaF,1)
	
	#Plotting
	rep=3
	plt.plot(x,t,color=(random.random(),random.random(),random.random()), marker='o', linestyle=' ')	
	plt.legend(["f(x)+e"],loc='lower left')	
	#plt.title("Neural Network (depth: "+ str(depth)+" h: "+str(h)+") vs. Gaussian Process")
	for i in range(rep):
		plt.plot(xp,gp.eval(x,t,xp,[],m),color=(random.random(),random.random(),random.random()), marker='', linestyle='-')	
	plt.legend(["samples"],loc='lower left')

	plt.xlabel('x')        
	plt.ylabel('y')

	plt.show()	
example1(n=50,k=50)
#example2(n=50,k=50)