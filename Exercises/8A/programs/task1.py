import numpy as np
import matplotlib.pyplot as plt 

class GibbsRegression:

	def __init__(self,w0,r0,x0,t0,Sigma0):
		self.W=w0
		self.R=r0
		self.x=x0
		self.t=t0
		self.alpha0=0.01
		self.beta0=0.01
		self.Sigma0=Sigma0
		self.Sigma0Inv=np.linalg.inv(Sigma0)
		
		
		#Building of design matrix X
		self.m=len(w0) #number of basis functions, counting the constant w_0
		self.X=np.ones(shape=(len(t0),self.m)) #design matrix
		self.populate_design_matrix()

		
	def basis_function(self,i,j):
		return self.x[i]**j

	def populate_design_matrix(self): #Defining the design matrix
		for i in range(len(self.t)):
			for j in range(self.m):
				self.X[i,j]=self.basis_function(i,j)
		
	def run(self,K,BurnIn):	
		self.WSamples=np.zeros(shape=(K-BurnIn,self.m))
		self.RSamples=np.zeros(shape=(K-BurnIn))
		
		for i in range(K):
			if i>=BurnIn:
				self.WSamples[i-BurnIn]=self.W#copy
				self.RSamples[i-BurnIn]=self.R
			self.updateWeights()
			self.updateR()
		
	def updateWeights(self):
		for j in range(len(self.W)):
			self.condWi(j)
			
	#berechne bedingte Wahrscheinlichkeit p(w_j|w_(-j),r0,t,X)
	def condWi(self,j):
		w_invj=np.delete(self.W,j)
		X_invj=np.delete(self.X,j,1)
		Sigma0_invj=np.delete(self.Sigma0Inv,j,1)
		var=1/(self.Sigma0Inv[j,j]+self.R*self.X[:,j].T.dot(self.X[:,j]))
		mu=var*(self.R*(self.X[:,j].T.dot(self.t)-w_invj.dot(X_invj.T).dot(self.X[:,j]))-Sigma0_invj[j,:].dot(w_invj.T))
		self.W[j]=(np.random.normal(mu,var,1))[0]
		#print(mu,var)
		
	def updateR(self):
		N=len(self.t)
		alphaN=self.alpha0+0.5*N
		betaN=self.beta0+0.5*(self.t-(self.X).dot(self.W)).T.dot(self.t-(self.X).dot(self.W))
		self.R = np.random.gamma(alphaN, 1.0/betaN, size=1)	
		
	

		
	
	
	
	

	
def Regression2D():
	#Hyperparameter
	w0=np.array([0.0,0.0,0.0])
	r0=1
	Sigma0=np.eye(3)
	
	x=np.random.uniform(0,2,20)
	t=[0.5*i*i+3.5*i+np.random.normal(0,0.1)+4 for i in x] #Beispielfunktion
	
	
	Sampler=GibbsRegression(w0,r0,x,t,Sigma0)
	Sampler.run(1000,20)
	plt.plot(Sampler.WSamples[:,0],Sampler.WSamples[:,1],marker='o',linestyle='-',color='r')
	plt.title("Path of the sampler after burn-in")
	plt.show()


	# Plot histogramm of weights and precision
	ax = plt.subplot(311)
	plt.hist(Sampler.WSamples[:,0])
	plt.title('intercept')
	plt.subplot(312, sharex=ax)
	plt.hist(Sampler.WSamples[:,1])
	plt.title('w_1')
	plt.subplot(313)	
	plt.hist(Sampler.WSamples[:,2])
	plt.title('w_2')
	plt.show()
	
	


#Hier wird ein Beispiel aufgerufen
Regression2D()