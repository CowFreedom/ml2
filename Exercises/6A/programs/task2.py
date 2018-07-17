from scipy.linalg import sqrtm
import numpy as np
import matplotlib.pyplot as plt 




def drawSamples(n,mean,cov):
	A=sqrtm(cov)
	print(A)
	sample=np.zeros(shape=(n,len(mean)))
	for i in range(n):
		sample[i]=A.dot(np.random.multivariate_normal(np.zeros(shape=(len(mean))),np.eye(len(mean))))+mean
	return sample
	
def plotsamples(n, mean, cov):
	if len(mean) !=2:
		print("Fehler: Dimension nicht 2")
		return 0

	samples=drawSamples(n,mean,cov)
	
	fig = plt.figure()
	ax2 = fig.add_subplot(111)
	ax2.plot(samples[:,0],samples[:,1],color='green', marker='o', linestyle=' ')
	plt.title("Samples")
	plt.xlabel('x')        
	plt.ylabel('y')
	plt.show()	
	
mean=np.array([2,-1])
cov=np.array([[4,2],[2,1]])

#mean=np.array([0,0])
#cov=np.array([[1,-1],[-1,1]]) 
plotsamples(20,mean,cov)
#samples=drawSamples(10,mean,cov) #samples have to be on the line x=-y
