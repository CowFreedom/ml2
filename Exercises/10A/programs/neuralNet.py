import numpy as np
import matplotlib.pyplot as plt 
import math
import random

#Output function f uses randomly initialized weights
#Weights do not get updated (no training)

class NeuralNet:

	def __init__(self,weightSigmas,hidden_layers,depth):
		self.sigmaStart=weightSigmas[0]
		self.sigmaStartBias=weightSigmas[1] #weights between hidden layers
		self.sigmaFunctionBias=weightSigmas[2] #weights of the output f
		self.sigmaFunction=weightSigmas[3] #weights of the output f
		self.h=hidden_layers
		self.depth=depth

	

	def initialize_weights(self):
		#randomly initialize weights of first layer
		self.startWeights=np.zeros(shape=(self.h,2))
		for i in range(self.h):
			self.startWeights[i]=[np.random.normal(0,self.sigmaStartBias),np.random.normal(0,self.sigmaStart)]
			
		#randomly initialize weights between hidden layers
		
		self.hiddenlayerWeightsBias=np.zeros(shape=(self.depth-1,self.h))
		for i in range(self.depth-1):
				self.hiddenlayerWeightsBias[i]=np.random.normal(0,self.sigmaStartBias,self.h)
				
		self.hiddenlayerWeights=np.zeros(shape=(self.depth-1,self.h,self.h))
		for i in range(self.depth-1):
			for j in range(self.h):
				self.hiddenlayerWeights[i,j]=np.random.normal(0,self.sigmaStart,self.h)
			
		#randomly initialize weights of output
		self.outputWeights=np.zeros(shape=(self.h))
		self.outputWeightsBias=np.random.normal(0,self.sigmaFunctionBias)
		for i in range(self.h):
			self.outputWeights[i]=np.random.normal(0,self.sigmaFunction)
		
	
	def g(self,x):
		#return np.tanh(x)	
		return math.erf(x)
			
	def eval(self,x):
		self.initialize_weights()
		output=np.zeros(len(x))
		prev=np.zeros(shape=(self.h))
		curr=np.zeros(shape=(self.h))
		
		for l in range(len(x)):
		
			for i in range(self.h):
				prev[i]=self.g(self.startWeights[i][0]+self.startWeights[i][1]*x[l])
			
			
			for k in range(self.depth-1):
				for i in range(self.h):	
					sum=self.hiddenlayerWeightsBias[k,i] #biasterm
					for j in range(self.h):
						sum=sum+self.hiddenlayerWeights[k,i,j]*prev[j]
					curr[i]=self.g(sum)
				prev=np.copy(curr)
			sum=self.outputWeightsBias
			
			sum=sum+self.outputWeights.dot(prev)
			
			output[l]=sum
		
		return output
		
class GP:

	def __init__(self,weightSigmas):
		self.sigmaStart=weightSigmas[0]
		self.sigmaStartBias=weightSigmas[1] #weights between hidden layers
		self.sigmaFunctionBias=weightSigmas[2] #weights of the output f
		self.sigmaFunction=weightSigmas[3] #weights of the output f
		
	def kernel(self,x,y):
		#return (self.sigmaFunction**2)*np.tanh(x)*np.tanh(y)+self.sigmaFunction**2
		nominator=2*(self.sigmaStartBias**2+(self.sigmaStart**2)*x*y)
		denominator=np.sqrt((1+2*(self.sigmaStartBias**2+(self.sigmaStart**2)*x*x))*(1+2*((self.sigmaStartBias**2+(self.sigmaStart**2)*y*y))))
		return (self.sigmaFunction**2)*(2/np.pi)*np.arcsin((nominator)/(denominator))+self.sigmaFunctionBias**2
		
	def calculate_cov(self,x):
		cov=np.zeros(shape=(len(x),len(x)))
	
		for i in range(len(x)):
			for j in range(len(x)):
				cov[i,j]=self.kernel(x[i],x[j])

		return cov
			
	def eval(self,x):
		mean=np.zeros(len(x))
		cov=self.calculate_cov(x)
		return np.random.multivariate_normal(mean,cov)

		

		
		
		
def OneHiddenLayerTest():
	rep=5 #number of plotted function per method
	n=200 #number of function evaluations
	h=400 #hidden layers per depth
	sigmaStart=8
	sigmaStartBias=4 #weights between hidden layers
	sigmaFunctionBias=0 #weights of the output f
	sigmaFunction= 0.5#weights of the output f

	depth=3 #depth of hidden layers (breadth of neural network)
	NN=NeuralNet((sigmaStart,sigmaStartBias,sigmaFunctionBias,(1/np.sqrt(h))*sigmaFunction),h,depth)
	x=np.linspace(-7,6,n)
	
	GaussianProcess=GP((sigmaStart,sigmaStartBias,sigmaFunctionBias,sigmaFunction))
	
	plt.subplot(2, 1, 1)
	for i in range(rep):
		plt.plot(x,NN.eval(x),color=(random.random(),random.random(),random.random()), marker='', linestyle='-',label='Random Perceptron')
	plt.legend(["NN"],loc='lower left')
	plt.xlabel('x')        
	plt.ylabel('y')
	plt.title("Neural Network (depth: "+ str(depth)+" h: "+str(h)+") vs. Gaussian Process")
	plt.subplot(2, 1, 2)
	for i in range(rep):
		plt.plot(x,GaussianProcess.eval(x),color=(random.random(),random.random(),random.random()), marker='', linestyle='-')	
	plt.legend(["GP"],loc='lower left')	
	plt.xlabel('x')        
	plt.ylabel('y')
	plt.show()
	

	
OneHiddenLayerTest()