# import numpy as np
# import scipy.stats as st    
# def MCMC_Metropolis(n,x,y):
#     xy = x+y
#     s_xy = sum(xy)
#     J = np.array([[1,0],[0,1],[-1,-1]])
#     b = np.array([[0],[0],[1]])
#     var0 = xy[0]*(s_xy-xy[0])/(s_xy**2*2*(s_xy+1))
#     var1 = xy[1]*(s_xy-xy[1])/(s_xy**2*2*(s_xy+1))
#     var2 = xy[2]*(s_xy-xy[2])/(s_xy**2*2*(s_xy+1))
#     cov01 = -xy[0]*xy[1]/(s_xy**2*2*(s_xy+1))
#     cov02 = -xy[0]*xy[2]/(s_xy**2*2*(s_xy+1))
#     cov12 = -xy[1]*xy[2]/(s_xy**2*2*(s_xy+1))
#     cov = np.array([[var0,cov01,cov02],[cov01,var1,cov12],[cov02,cov12,var2]])
#     cov_1 = np.linalg.inv(cov)
#     S_1 = np.dot(np.dot(np.transpose(J),cov_1),J)
#     S = np.linalg.inv(S_1)
#     print(S)
#     mean = np.transpose(np.dot(np.dot(np.dot(S,np.transpose(J)),cov_1),b))
#     mean = mean[0]
#     print(mean)
#     # cov = []
#     # for i in range(len(xy)):
#     #     linha = []
#     #     for j in range(len(xy)):
#     #         if i == j:
#     #             aux = xy[i]*(s_xy-xy[i])/(s_xy**2*(s_xy+1))
#     #         else:
#     #             aux = -xy[i]*xy[j]/(s_xy**2*(s_xy+1))
#     #         linha.append(aux)
#     #     cov.append(linha)
#     # cov = np.array(cov)
#     theta_old = np.array([0.3,0.25,0.45])
#     burn_in = 1000
#     sample = np.zeros((n+burn_in,3))
#     # t = 0
#     for i in range(burn_in+n):
#         sample[i] = theta_old
#         # mean = theta_old
#         c = np.random.multivariate_normal(mean, S)
#         print(c)
#         a = np.dot(J,np.transpose(c))+np.transpose(b)
#         a = a[0]
#         # print(aux)
#         # theta_star = np.zeros(3)
#         theta_star = theta_old+a
#         print(theta_star)
#         # print(theta_star)
#         alpha = np.min([st.dirichlet.pdf(theta_star,xy)/st.dirichlet.pdf(theta_old,xy),1])
#         u = np.random.uniform()
#         if u <= alpha:
#             theta_old = theta_star
#     sample = sample[burn_in:]
#     return sample

    

# cov = [[2,-0.86,-0.15],[-0.86,3.4,0.48],[-0.15,0.48,0.82]]
# mean = [2,1,3]
# x = np.random.multivariate_normal(mean, cov)
# print(x)
# x = np.zeros([5,2])
# x[1] = [1,2]
# print(x)
from scipy.special import gamma
import numpy as np
import statistics
import scipy.stats as st
from time import time
import matplotlib.pyplot as plt

def f(theta, params):
    
    if theta[0] < 0 or theta[1] < 0 or theta[2] < 0:
        return -1

    beta = np.prod(gamma(params))/ gamma(np.sum(params))
    
    resul = (1/beta)*np.prod(theta**(params - 1))
    
    return resul

def MCMC_Metropolis(n,x,y):
    
    t1 = time()
    
    xy = x+y
    s_xy = sum(xy)
    var0 = xy[0]*(s_xy-xy[0])/(s_xy**2*(s_xy+1))
    var1 = xy[1]*(s_xy-xy[1])/(s_xy**2*(s_xy+1))
    var2 = xy[2]*(s_xy-xy[2])/(s_xy**2*(s_xy+1))
    cov01 = -xy[0]*xy[1]/(s_xy**2*2*(s_xy+1))
    cov02 = -xy[0]*xy[2]/(s_xy**2*2*(s_xy+1))
    cov12 = -xy[1]*xy[2]/(s_xy**2*2*(s_xy+1))
    cov = 0.1*np.array([[var0,cov01,cov02],[cov01,var1,cov12],[cov02,cov12,var2]])

    # cov = []
    # for i in range(len(xy)):
    #     linha = []
    #     for j in range(len(xy)):
    #         if i == j:
    #             aux = xy[i]*(s_xy-xy[i])/(s_xy**2*(s_xy+1))
    #         else:
    #             aux = -xy[i]*xy[j]/(s_xy**2*(s_xy+1))
    #         linha.append(aux)
    #     cov.append(linha)
    # cov = np.array(cov)
    theta_old = np.array([0.01,0.05,0.94])
    burn_in = 0
    # mean = np.array([0,0,0])
    sample = np.zeros((n+burn_in,3))
    # t = 0
    for i in range(burn_in+n):
        mean = theta_old
        sample[i] = theta_old
        # print(aux)
        x_star, y_star, z = np.random.multivariate_normal(mean, cov)
        z_star = 1 - (x_star + y_star)
        theta_star = np.array([x_star, y_star, z_star])
        alpha = np.min([f(theta_star,xy)/f(theta_old,xy),1])

        u = np.random.uniform()
        if u <= alpha:
            theta_old = theta_star
            
    # sample = sample[burn_in:]
    
    t2 = time()
    
    return sample, t2 - t1
def thetamcmc(n,x,y):
    
    t1 = time()
    
    xy = x+y
    s_xy = sum(xy)
    var0 = xy[0]*(s_xy-xy[0])/(s_xy**2*(s_xy+1))
    var1 = xy[1]*(s_xy-xy[1])/(s_xy**2*(s_xy+1))
    var2 = xy[2]*(s_xy-xy[2])/(s_xy**2*(s_xy+1))
    cov01 = -xy[0]*xy[1]/(s_xy**2*2*(s_xy+1))
    cov02 = -xy[0]*xy[2]/(s_xy**2*2*(s_xy+1))
    cov12 = -xy[1]*xy[2]/(s_xy**2*2*(s_xy+1))
    cov = np.array([[var0,cov01,cov02],[cov01,var1,cov12],[cov02,cov12,var2]])
    theta_old = np.array([0.3,0.25,0.45])
    burn_in = 0
    # mean = np.array([0,0,0])
    sample = np.zeros((n+burn_in,3))
    sample[0] = theta_old
    i = 0
    j=0
    while i < burn_in+n:
        mean = theta_old
        # sample[i] = theta_old
        x_star, y_star, z = np.random.multivariate_normal(mean, cov)
        z_star = 1 - (x_star + y_star)
        theta_star = np.array([x_star, y_star, z_star])
        alpha = np.min([f(theta_star,xy)/f(theta_old,xy),1])

        u = np.random.uniform()
        if u <= alpha:
            theta_old = theta_star
            sample[i] = theta_old
            i = i+1
        j+=1  
    sample = sample[burn_in:]
    
    t2 = time()
    return sample,i,j,t2-t1
vetor_x = np.array([1,2,3])
vetor_y = np.array([2,1,3])
# s,i,j,t = thetamcmc(500,vetor_x,vetor_y)
s,t = MCMC_Metropolis(500,vetor_x,vetor_y)

# print(len(s))
s1 = np.transpose(s)
# print(i/j)
print(t)
theta1 = s1[0]
theta2 = s1[1]
theta3 = s1[2]
# print(theta3)
# time = np.linspace(1,500,500)
# plt.figure(1)
# plt.plot(time,theta1)
# plt.xlabel('iteração')
# plt.ylabel('Valor theta1')
# plt.grid()
# plt.figure(2)
# plt.plot(time,theta2)
# plt.xlabel('iteração')
# plt.ylabel('Valor theta2')
# plt.grid()
# plt.figure(3)
# plt.plot(time, theta3)
# plt.xlabel('iteração')
# plt.ylabel('Valor theta3')
# plt.grid()
# plt.show()
