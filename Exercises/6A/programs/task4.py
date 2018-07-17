import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal as mvn
from mpl_toolkits.mplot3d import Axes3D

#Der Code ist sehr auf Normalverteilungen zugeschnitten.
#Um aus anderen Verteilungen Stichproben zu ziehen,
#sollte er neu geschrieben werden.

def drawSamples(n,mean,cov,c):
	#proposal function
	samples=np.zeros(shape=(n,len(mean)))
	hits=0
	iter=0
	alpha=1.01 #scaling of the umbrella distribution
	c=c #has to be larger than 1
	while (hits <n and iter<=6000):
		x=np.random.multivariate_normal(mean, alpha*cov)
		u=np.random.uniform(0,c*mvn.pdf(x,mean,alpha*cov),1)[0]
		if iter==0:
			print("Rate:",mvn.pdf(mean,mean, cov)/mvn.pdf(mean,mean,alpha*cov))
			print("target:",mvn.pdf(x,mean, cov),"overlap:",mvn.pdf(x,mean,alpha*cov))
		if (u<=mvn.pdf(x,mean,cov)):
			samples[hits]=x
			hits=hits+1
		iter=iter+1
	
	print("Hitrate: "+str(hits/iter))
	return (samples,hits/iter)

def sampleMean(arr):
	return np.mean(arr,axis=0)
		

	
def plotsampledGaussian2d(n,c):
	dimension=2
	N=n
	cov=0.1*np.eye(dimension)
	mu=np.zeros(dimension)
	mean=np.array([0,0])
	 #overlap factor
	samples=drawSamples(N,mean,cov,c)[0]
	#realSamples=np.random.multivariate_normal(mu, sigma,N)
	#Plotting
	fig = plt.figure()
	ax2 = fig.add_subplot(111)
	
	# Generate grid points
	x, y = np.meshgrid(np.linspace(-1,2,100),np.linspace(-1,2,100))
	xy = np.column_stack([x.flat, y.flat])
	
	overlap=[c*t for t in mvn.pdf(xy, mean, 1.01*cov).reshape(x.shape)]

	# density values at the grid points
	Z = mvn.pdf(xy, mean, cov).reshape(x.shape)
	#overlap = mvn.pdf(xy, mean, ovfactor*cov).reshape(x.shape)
	# arbitrary contour levels
	contour_level = [0.1,0.2,0.3]

	fig = plt.contour(x, y, Z, levels = contour_level,colors='g')
	fig = plt.contour(x, y, overlap, levels = contour_level,colors='r')
	plt.title("Normal distributions with overlap")
	plt.xlabel('x')        
	plt.ylabel('y')
	plt.show()

def plotsampledGaussians3d(n,c):
	dimension=2
	N=n
	cov=1*np.eye(dimension)
	mean=np.array([0,0])
	
	samples=drawSamples(N,mean,cov,c)[0]

	
	#Create grid and multivariate normal
	x = np.linspace(-10,10,500)
	y = np.linspace(-10,10,500)
	X, Y = np.meshgrid(x,y)
	pos = np.empty(X.shape + (2,))
	pos[:, :, 0] = X; pos[:, :, 1] = Y
	rv = mvn(mean, cov)
	overlap=lambda pos:c*mvn(mean, cov).pdf(pos)
	#Make a 3D plot
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
	ax.plot_wireframe(X, Y, overlap(pos),cmap='viridis',linewidth=0.5)
	ax.set_xlabel('X axis')
	ax.set_ylabel('Y axis')
	ax.set_zlabel('Z axis')
	plt.show()

def plotRejectionRate(n,c,dimension):
	#Distribution parameters of target distribution

	dim=np.zeros(shape=(dimension))
	rates=np.zeros(shape=(dimension))
	
	for i in range(dimension):
		cov=1*np.eye(i+1)
		mean=np.zeros(shape=(i+1))
		rates[i]=drawSamples(n,mean,cov,c)[1]
		dim[i]=i+1
		
	fig = plt.figure()
	ax2 = fig.add_subplot(111)
	ax2.plot(dim,rates,color='green', marker='o', linestyle='-')
	plt.title("scaling coefficent c:"+str(c))
	plt.xlabel('dimension')        
	plt.ylabel('hitrate')
	plt.show()
		
def importance_sampling(n,mean,cov,dimension):
	samples=np.zeros(shape=(n))
	weights=np.zeros(shape=(n))
	iter=0
	alpha=1.01 #scaling of the umbrella distribution Q
	while iter <n:
		x=np.random.multivariate_normal(mean, alpha*cov) #draw from umbrella distribution
		samples[iter]=mvn.pdf(x,mean, cov)
		weights[iter]=mvn.pdf(x,mean, cov)/mvn.pdf(x,mean, alpha*cov)
		iter=iter+1

	#evaluate expectation

	sum_below=0
	for i in range(n):
		sum_below=sum_below+weights[i]
	
	sum_above=0
	for i in range(n):
		sum_above=samples[i]*weights[i]
	return (sum_above/sum_below,samples)

def plotIntegral_Importance_Sampling(n,dimension):
	dim=np.zeros(shape=(dimension))
	e_x=np.zeros(shape=(dimension))
	
	for i in range(dimension):
		cov=1*np.eye(i+1)
		mean=np.zeros(shape=(i+1))
		e_x[i]=importance_sampling(n,mean,cov,dimension)[0]
		dim[i]=i+1
		
	fig = plt.figure()
	ax2 = fig.add_subplot(111)
	ax2.plot(dim,e_x,color='green', marker='o', linestyle='-')
	plt.title("Value of integral")
	plt.xlabel('dimension')        
	plt.ylabel('E[f(x))')
	plt.show()

	

	
#plotsampledGaussian2d(10,2) #the first parameter gives the number of samples and the second one scales the overlap distribution (higher c equals leads to lower hit rate in rejection sampling)

#plotsampledGaussians3d(10,2)

plotRejectionRate(100,1.2,20)
#plotIntegral_Importance_Sampling(10,20)

#c=1
#dimension=500
#alpha=1.01
#for i in range(dimension):
#	cov=0.3*np.eye(i+1)
#	mean=np.zeros(shape=(i+1))
#	ratio=mvn.pdf(mean+1,mean,cov)/(c*mvn.pdf(mean+1,mean,alpha*cov))
#	print(ratio)

