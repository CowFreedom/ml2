import numpy as np
import matplotlib
matplotlib.use('PDF') 
import matplotlib.pyplot as plt 
from scipy.stats import beta as Beta

i=9
n=10
alpha=5
beta=5

samples=np.random.choice(2, n, replace=True, p=[0.3,0.7])
k=len([y for y in samples if y==1])

#x-axis values
x=np.linspace(0,1, 100)

#r'$\alpha=1, \beta$=1'

plt.title("alpha="+str(alpha)+", β="+str(beta)+", n="+str(n))
#prior
plt.plot(x,Beta.pdf(x,alpha,beta),'b-')
#posterior
plt.plot(x,Beta.pdf(x,alpha+k,beta+(n-k)),'r-')
plt.xlabel('θ')        
plt.ylabel('f(θ)')
#plt.show()

plt.savefig('C://Users//Tristan_local//Desktop//Figure_'+str(i)+'.png', format='png',alpha='0')