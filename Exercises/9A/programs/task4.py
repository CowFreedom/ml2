import numpy as np
import matplotlib.pyplot as plt 


class randomExplicitFunctions:

	def __init__(self):
		print("Start")
	
#k=number of random functions, #x=sample points, m=number basis functions #r0=noise of weights
	def sample_from_polynomials(self,k,x,m,r0):
		self.r0=r0*np.eye(m)
		self.mu=np.zeros(m)
		n=len(x)
		output=np.zeros(shape=(k,n))
		
		for f in range(k):
			w=np.random.multivariate_normal(self.mu, self.r0)
			for i in range(n):
				for l in range(m):
					output[f][i]=output[f][i]+w[l]*(x[i]**l)
		return output
		
#k=number of random functions, #x=sample points, mu=means of basis functions #r0=noise of weights
	def sample_from_exponentials(self,k,x,mu,r0):
		self.r0=r0*np.eye(len(mu))
		n=len(x)
		output=np.zeros(shape=(k,n))
		
		for f in range(k):
			w=np.random.multivariate_normal(np.zeros(len(mu)), self.r0)
			for i in range(n):
				for l in range(len(mu)):
					output[f][i]=output[f][i]+w[l]*np.exp(-0.5*(x[i]-mu[l])**2)
		return output		
		

class randomKernelFunctions:
	def __init__(self,x):
		self.data=x
		print("Kernels initiiert, Nigga!")
		
	def createGramMatrix(self,f,args):
		self.GramMatrix=np.eye(len(self.data))
		
		for i in range(len(self.data)):
			for j in range(len(self.data)):
				self.GramMatrix[i,j]=f(self.data[i],self.data[j],args)
	
	#x=datapoint 1, y=datapoint 2, sigma=std deviation
	def LinearKernel(self,x,y,args):
		sigma=args
		return sigma*sigma*x*y
		
	def CubicKernel(self,x,y,args):
		sigma=args
		return sigma*sigma*((x*y+1)**3)
		
	def ExponentialKernel(self,x,y,args):
		sigma=args[0]
		l=args[1]
		return sigma*sigma*np.exp(-(1/(2*l*l))*(x-y)**2)		
		
	def sample(self,k,selector,args):
		options = {0 : self.LinearKernel,
				   1 : self.CubicKernel,
				   2 : self.ExponentialKernel,
		}	
		self.createGramMatrix(options[selector],args)
		
		output=np.random.multivariate_normal(np.zeros(len(self.data)), self.GramMatrix,k)
		
	
			#output[f]=np.random.multivariate_normal(np.zeros(len(self.data)), self.GramMatrix,len(self.data))
			#print("F",np.random.multivariate_normal(np.zeros(len(self.data)), self.GramMatrix,len(self.data)))
		return output			

	
def draw_explicit_polynomials():
	sampler=randomExplicitFunctions()
	k=5 #number of functions
	r0=1
	data=np.linspace(-1,1,100)
	samples=sampler.sample_from_polynomials(k,data,3,r0)
	print(samples[0])
	for i in range(k):		
		plt.plot(data,samples[i],color=(np.random.rand(),np.random.rand(),np.random.rand()), marker='', linestyle='-')
		plt.xlabel('x')        
		plt.ylabel('f(x)')
	plt.title("Explicit samples from polynomials")			
	plt.show()
	
def draw_explicit_exponentials():
	sampler=randomExplicitFunctions()
	k=5#number of functions
	r0=1
	data=np.linspace(-1,1,100)
	mu=np.array(np.linspace(-1,1,3))#change the means

	samples=sampler.sample_from_exponentials(k,data,mu,r0)
	for i in range(k):		
		plt.plot(data,samples[i],color=(np.random.rand(),np.random.rand(),np.random.rand()), marker='', linestyle='-')
		plt.xlabel('x')        
		plt.ylabel('f(x)')
	plt.title("Explicit samples from exponentials")	
	plt.show()

def draw_kernelFunctions():
	data=np.linspace(-1,1,100)
	sampler=randomKernelFunctions(data)
	k=2#number of functions
	#samples=sampler.sample(k,0,1)
	L=1
	samples=sampler.sample(k,2,(1,L))
	print(samples)
	for i in range(k):		
		plt.plot(data,samples[i], marker='', linestyle='-',color=(np.random.rand(),np.random.rand(),np.random.rand()))
		plt.xlabel('x')        
		plt.ylabel('f(x)')
		
	plt.title("Explicit samples from Kernel")	
	plt.show()
		
	
draw_explicit_polynomials()
draw_explicit_exponentials()
draw_kernelFunctions()

