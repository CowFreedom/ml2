import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal
import copy
from scipy.stats import mvn

def loadData(path):
	return np.loadtxt(path,delimiter='  ')

class expectation_maximization:
	def __init__(self,data,_k):
		self.data=data
		self.k=_k
		self.data_dimension=self.data.shape[1]
		self.len_data=self.data.shape[0]
		max_values=np.amax(self.data,axis=0)
		min_values=np.amin(self.data,axis=0)
		self.mu=np.zeros(shape=(_k,self.data_dimension))
		self.sigma=np.zeros(shape=(_k,self.data_dimension,self.data_dimension))
		#initialize mixing coefficients uniformly
		self.mixing_coefficients=(1/_k)*np.ones(_k)
		#initialize cluster centers randomly
		for i in range(_k):
			for j in range(self.data_dimension):
				self.mu[i,j]=np.random.uniform(min_values[j],max_values[j],1)
				
		#initialize covariances
		epsilon=0.1 #PlAY
		for i in range(_k):
			self.sigma[i]=epsilon*np.eye(self.data_dimension)
			
		#initialize responsibilities
		self.responsibilities=np.zeros(shape=(self.len_data,_k))
		self.log_likelihood_old=0
		self.log_likelihood_new=self.log_likelihood()
		self.run_algorithm()
			
	def log_likelihood(self):
		#print("Bin drin")
		sum=0
		for i in range(self.len_data):
			temp=0
			for j in range(self.k):
				temp=temp+self.mixing_coefficients[j]*multivariate_normal.pdf(self.data[i],self.mu[j],self.sigma[j])
			sum=sum+np.log(temp)
		return sum
	
	#p(x|Class)
	def conditional_likelihood(self,data,Z):
		return multivariate_normal.pdf(data,self.mu[Z],self.sigma[Z])
		#return multivariate_normal.cdf(data+0.01,self.mu[Z],self.sigma[Z])-multivariate_normal.cdf(data-0.01,self.mu[Z],self.sigma[Z])
		#return mvn.mvnun(data-0.5,data+0.5,self.mu[Z],self.sigma[Z])[0]
	
	def calc_responsibilities(self):
		for i in range(self.len_data):
			evidence=0
			for j in range(self.k):
				evidence=evidence+self.mixing_coefficients[j]*multivariate_normal.pdf(self.data[i],self.mu[j],self.sigma[j])
			for j in range(self.k):
				joint=self.mixing_coefficients[j]*multivariate_normal.pdf(self.data[i],self.mu[j],self.sigma[j])
				self.responsibilities[i,j]=joint/evidence
				
				
	def hasConverged(self):
		if (abs(self.log_likelihood_old-self.log_likelihood_new) <1):
			return True
		else:
			return False
	
	def Nk(self,m):
		sum=0
		for i in range(self.len_data):
			sum=sum+self.responsibilities[i,m]
		return sum
	
	def calculate_mu(self):
		for j in range(self.k):
			temp=np.zeros(shape=(self.data_dimension))
			for i in range(self.len_data):
				temp=temp+self.responsibilities[i,j]*self.data[i]
			self.mu[j]=(1/self.Nk(j))*temp
			
	def calculate_sigma(self):
		for j in range(self.k):
			temp=np.zeros(shape=(self.data_dimension,self.data_dimension))
			for i in range(self.len_data):
				vec=(self.data[i]-self.mu[j]).reshape(self.data_dimension,1)
				temp=temp+self.responsibilities[i,j]*vec.dot(vec.transpose())
				#print("vec:",vec)
				#print("mat:",temp)
			self.sigma[j]=(1/self.Nk(j))*temp
			
	def calculate_mixing_coefficients(self):
		for i in range(self.k):
			self.mixing_coefficients[i]=(self.Nk(i))/self.len_data
			
	def CheckVariance(self):
		for i in range(self.k):
			for j in range(self.data_dimension):
				if (self.sigma[i,j,j]<0.1): #PlAY
					return False
				else:
					return True
			
	
	def run_algorithm(self):
		iter=0
		while (self.hasConverged()==False or self.CheckVariance()):
			self.log_likelihood_old=self.log_likelihood_new
			#Expectation step:
			self.calc_responsibilities()
			
			#Maximization step:
			self.calculate_mu()

			self.calculate_sigma()
			self.calculate_mixing_coefficients()
			#print(self.sigma)
			self.log_likelihood_new=self.log_likelihood()
			print(iter)

			iter=iter+1
			
class k_means:
	def __init__(self,data,_k):
		self.data=data
		self.k=_k
		self.data_dimension=self.data.shape[1]
		self.len_data=self.data.shape[0]
		max_values=np.amax(self.data,axis=0)
		min_values=np.amin(self.data,axis=0)
		self.mu_old=np.zeros(shape=(_k,self.data_dimension))
		self.mu=np.zeros(shape=(_k,self.data_dimension))
		self.labels=np.zeros(shape=(self.len_data))
		#initialize cluster centers randomly
		for i in range(_k):
			for j in range(self.data_dimension):
				self.mu[i,j]=np.random.uniform(min_values[j],max_values[j],1)	
		self.run_algorithm()
			
	def E_Step(self):
			for i in range(self.len_data):
				current_min=np.linalg.norm(self.data[i]-self.mu[0],2)
				classlabel=0
				for j in range(self.k -1):
					current_dist=np.linalg.norm(self.data[i]-self.mu[j+1],2)
					if(current_dist<current_min):
						current_min=current_dist
						classlabel=j+1
				self.labels[i]=classlabel
				
	def M_Step(self):
		for i in range(self.len_data):
			current_label=int(self.labels[i])
			self.mu[current_label]=self.mu[current_label]+self.data[i]
		for i in range(self.k):
			if(np.count_nonzero(self.labels==i)!= 0):
				self.mu[i]=(1/np.count_nonzero(self.labels==i))*self.mu[i]
			else:
				self.mu[i]=0*self.mu[i]
	
	def hasConverged(self):
		if (np.linalg.norm(self.mu_old-self.mu,2)<0.01):
			return True
		else:
			return False
	
	def run_algorithm(self):
		iter=0
		while(self.hasConverged()==False and iter<100):
			print(iter)
			self.mu_old=copy.deepcopy(self.mu) #kopieren weil sonst ist mu_old nur ein Pointer zu mu
			self.E_Step()
			self.M_Step()
			iter=iter+1

		
	
		
#Main
path="YOUR_PATH/5A/data/data.txt"	 #CHANGE PATH ACCORDING TO YOUR SYSTEM PLEASE	
data=loadData(path)


def plot_data():
	fig = plt.figure()
	ax2 = fig.add_subplot(111)
	ax2.plot(data[:,0],data[:,1],color='green', marker='o', linestyle='',label='data')
	plt.legend(loc='lower left');
	plt.xlabel('x')        
	plt.ylabel('y')
	plt.show()



def plot_EM():
	k=2 #number of cluster centers
	em=expectation_maximization(data,k)
	
	fig = plt.figure()
	ax2 = fig.add_subplot(111)
	ax2.plot(data[:,0],data[:,1],color='green', marker='o', linestyle='',label='data')
	plt.legend(loc='lower left');
	plt.xlabel('x')        
	plt.ylabel('y')
	ax2.plot(em.mu[:,0],em.mu[:,1],color='red', marker='x', linestyle='',label='cluster centers')
	plt.title("Clustering through Expectation Maximization")
	plt.show()

#p(Datapoint|Class) where Z=Class
def plot_EM_conditionals(Z):
	k=2 #number of cluster centers
	if (Z>=k):
		print("Error: We have not partioned"+str(Z)+"classes.")
	em=expectation_maximization(data,k)
	
	fig = plt.figure()
	ax2 = fig.add_subplot(111)
	
	maxdensity=max([em.conditional_likelihood(data,Z) for data in em.data])	
	for i in range(em.len_data):
		#print(float(em.conditional_likelihood(data[i],Z)))
		ax2.plot(data[i,0],data[i,1],color=(0,0,em.conditional_likelihood(data[i],Z)/maxdensity), marker='o', linestyle='')
	print(em.sigma)
	print(em.mu)
	plt.legend(loc='lower left');
	plt.xlabel('x')        
	plt.ylabel('y')
	ax2.plot(em.mu[:,0],em.mu[:,1],color='red', marker='x', linestyle='',label='cluster centers')
	plt.title("P(Datapoint|Class "+str(Z)+")")
	plt.show()	
	

def plot_k_means():

	k=2 #number of cluster centers
	kmeans=k_means(data,k)		
	fig = plt.figure()
	ax2 = fig.add_subplot(111)
	#print(kmeans.labels )
	#print(kmeans.mu)
	numberofClasses=max(kmeans.labels)+1
	for i in range(kmeans.len_data):
		#print(float(kmeans.labels[i]/numberofClasses))
		ax2.plot(data[i,0],data[i,1],color=(float(kmeans.labels[i]/numberofClasses),float(kmeans.labels[i]/numberofClasses),float(kmeans.labels[i]/numberofClasses)), marker='o', linestyle='')
	
	#ax2.plot(data[:,0],data[:,1],color='green', marker='o', linestyle='',label='data')

	plt.xlabel('x')        
	plt.ylabel('y')
	ax2.plot(kmeans.mu[:,0],kmeans.mu[:,1],color='red', marker='x', linestyle='',label='cluster centers')
	plt.legend(loc='lower left');
	plt.title(str(k)+" means")
	plt.show()	
	
#plot_EM()
plot_EM_conditionals(0)

#plot_k_means()