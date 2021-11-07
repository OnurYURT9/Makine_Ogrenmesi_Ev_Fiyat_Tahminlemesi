# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 15:47:10 2021

@author: ONUR Yurt
181312051
"""
import numpy as np  
from matplotlib import pyplot as plt   
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#part1 : load the data from file
data = np.loadtxt('population_vs_profil.data')
N=data.shape[0]


#part2 : plot the data in 2d as population vs profit

plt.figure(1)
X=data[:,0];
Y=data[:,1];
plt.plot(X,Y,'.r',label='Training data')
plt.xlabel('City population in $10.000s')
plt.ylabel('Profit in $10.000s')

#part3 : Implement the gradient step calculation for theta

def computeStep(X,y,theta):
    dist= np.dot(X,theta)-y
    step0=sum(dist*X[:,0])/len(y)
    step1=sum(dist*X[:,1])/len(y)
    return np.array([step0,step1])

#part4 :  Implement the cost function calculation

def computeCost(X,y,theta):
    dist= np.dot(X,theta)-y
    dist_sq = np.power(dist,2) #kare alma
    cost = np.mean(dist_sq)/2
    return cost

#part5 : Prepare the data so that the input X has two columns: first a column of ones to accomodate theta0 and then a column of city population data

X=np.vstack((np.ones(N),X).T
            
# Part 6: Apply linear regression with gradient descent
theta = np.array([0,0])    
num_iter = 1500
alpha_line = [[0.1, '-b'], [0.03, '-r'], [0.01, '-g'], [0.003, ':b'], [0.001, ':r'], [0.0003, ':g']]
init_cost=computeCost(X,y,theta) 
print ('The initial cost is %f.' %init_cost)

plt.figure()
plt.ylim(0,100)
plt.xlim(0,10)
final_theta = []
for alpha, line in alpha_line:
    J_history = []
    theta = np.array([0,0])
    for i in range(num_iter):
        theta = theta-alpha*computeStp(X,y,theta)
        J_history.append(computeCost(X,y,theta))
    plt.plot(J_history, line, linewidth=3, label='alpha:%5.4f'%alpha)
    final_theta.append(theta)
    print ('Final cost after %d iterations is %f.' %(num_iter, J_history[-1]))
plt.legend(fontsize=12)

# Part 7: Plot the resulting line and make predictions with the best performing theta 
plt.figure(1)
plt.plot(X[:,1],np.dot(X,best_theta),'-', label='Linear regression with gradient descent')
plt.scatter(x,y)
plt.plot(a,m*a+b)
plot.show()

print ('Estimated profit for a city of population 35000 is % 7.2f.' %y1)
print ('Estimated profit for a city of population 70000 is % 7.2f.' %y2)

# Part 8: Plot cost function as a 2D surface over theta0 and theta1 axes
grid_size = 200
theta0_vals = np.linspace(-10, 10, grid_size)
theta1_vals = np.linspace(-1, 4, grid_size)
theta0, theta1 = np.meshgrid(theta0_vals, theta1_vals)
cost_2d = np.zeros((grid_size,grid_size))
for t0 in range(grid_size):
    for t1 in range(grid_size):
        theta = [theta0[t0,t1],theta1[t0,t1]]
        cost_2d[t0,t1] = computeCost(X,y,theta)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta0, theta1, cost_2d, cmap=cm.jet, linewidth=0, antialiased=False, alpha=0.5 )
ax.set_xlabel('Theta 0')
ax.set_ylabel('Theta 1')
ax.set_zlabel('Cost')
plt.figure()
plt.contour(theta0, theta1, cost_2d, 100)
plt.plot(best_theta[0], best_theta[1], 'xr')
plt.xlabel('Theta 0')
plt.ylabel('Theta 1')














