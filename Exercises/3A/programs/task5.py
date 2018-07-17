import numpy as np
import matplotlib.pyplot as plt 

#interval length [a,b) (a=inclusive, =exclusive)
a=0
b=1

#number of samples
n=25

#drawing n samples in [a,b)
x=np.random.uniform(a,b,n)

#dataset generating function
f_x=lambda x_0:np.sin(2*np.pi*x_0)+0.3*np.random.normal(0,1)

#target values of the data
t=f_x(x)

#Building of design matrix X
m=10 #number of basis functions, counting the constant w_0

X=np.ones(shape=(n,m)) #design matrix

def basis_function(x_0,mu):
	k=20 #affects locality of gaussians
	return np.exp(-0.5*k*(x_0-mu)*(x_0-mu))

def populate_design_matrix(): #Defining the design matrix
	i_p=(b-a)/(m)
	for i in range(n):
		for j in range(1,m):
			X[i,j]=basis_function(x[i],a+j*i_p)
			
populate_design_matrix()

#Posterior distribution
r_0=1 #prior precision
r_e=1/0.0009 #likelihood precision


Sigma_N=np.linalg.pinv(r_0*np.identity(m)+r_e*(X.transpose().dot(X)))
Mu_N=r_e*Sigma_N.dot(X.transpose()).dot(t)

#plotting results
repetitions=5

xAxis=np.linspace(a,b)

fig = plt.figure()
ax2 = fig.add_subplot(111)
ax2.plot(x,t,color='green', marker='o', linestyle='',label='data')
plt.legend(loc='lower left');
plt.xlabel('x')        
plt.ylabel('t')

#drawn weights
for k in range(repetitions):
	w=np.random.multivariate_normal(Mu_N, Sigma_N)

	#plotting the polynom

	def y_x(x_0):
		sum=w[0]
		i_p=(b-a)/(m)
		for i in range(1,m):
			print("w["+str(i)+"]:",w[i])
			#print("a+i_p:"+str(a+i*i_p))
			print("x0:"+str(x_0))
			print("Radial x0:"+str(basis_function(x_0,a+i*i_p)))
			print("Result:"+str(sum+basis_function(x_0,a+i*i_p)*w[i]))
			sum=sum+basis_function(x_0,a+i*i_p)*w[i]
		return sum
		
	y_values=[y_x(x0) for x0 in xAxis]
	ax2.plot(xAxis,y_values,label='y(x)')
plt.title("n="+str(n)+", "+str(m)+" basis functions")
plt.show()
