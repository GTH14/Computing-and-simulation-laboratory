from scipy.special import gamma
import numpy as np
import statistics
import scipy.stats as st
from time import time
def MCMC_Metropolis(n,x,y):
    t1 = time()
    xy = x+y
    s_xy = sum(xy)
    var0 = xy[0]*(s_xy-xy[0])/(s_xy**2*2*(s_xy+1))
    var1 = xy[1]*(s_xy-xy[1])/(s_xy**2*2*(s_xy+1))
    var2 = xy[2]*(s_xy-xy[2])/(s_xy**2*2*(s_xy+1))
    cov01 = -xy[0]*xy[1]/(s_xy**2*2*(s_xy+1))
    cov02 = -xy[0]*xy[2]/(s_xy**2*2*(s_xy+1))
    cov12 = -xy[1]*xy[2]/(s_xy**2*2*(s_xy+1))
    cov = np.array([[var0,cov01,cov02],[cov01,var1,cov12],[cov02,cov12,var2]])
    mean = np.array([0,0,0])
    burn_in = 1000
    theta_star = np.random.multivariate_normal(mean, cov, size = n+burn_in)
    theta_star = theta_star
    print(len(theta_star))
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
    theta_old = np.array([0.3,0.25,0.45])
    burn_in = 1000
    # mean = np.array([0,0,0])
    cont = 0
    sample = np.zeros((n+burn_in,3))
    # t = 0
    for i in range(burn_in+n):
        mean = theta_old
        sample[i] = theta_old
        # print(aux)
        while True:
            try:
                theta_star = np.random.multivariate_normal(mean, cov)
                alpha = np.min([st.dirichlet.pdf(theta_star,xy)/st.dirichlet.pdf(theta_old,xy),1])
                cont+=1
                break

            except:
                pass
        u = np.random.uniform()
        if u <= alpha:
            theta_old = theta_star
    sample = sample[burn_in:]
    t2 = time()
    return sample, t2-t1, cont

vetor_x = np.array([1,2,3])
vetor_y = np.array([2,1,3])
s,t, cont = MCMC_Metropolis(100000,vetor_x,vetor_y)
print(t)
print(cont)
# n = 10000
# burn_in = 10000
# cov = [[0.2,-0.086,-0.015],[-0.086,0.34,0.048],[-0.015,0.048,0.082]]
# mean = [0,0,0]
# theta_star = np.random.multivariate_normal(mean, cov, size = n+burn_in)
# print(len(theta_star))
# # theta_star = theta_star[theta_star[1]>=0]
# theta_star = theta_star[np.sum(theta_star,axis = 1) <=2 ]
# print(len(theta_star))