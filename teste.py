import scipy.stats as st
import numpy as np

def thetamcmc(n, vetor_x, vetor_y):

    # a soma dos vetores são os parametros para a dirichlet
    alpha = vetor_x + vetor_y
    print(alpha)

    den = (np.sum(alpha)**2)*(np.sum(alpha) + 1)
    
    var_theta1 = ((alpha[0])*(np.sum(alpha) - alpha[0]))/den
    var_theta2 = ((alpha[1])*(np.sum(alpha) - alpha[1]))/den
    var_theta3 = ((alpha[2])*(np.sum(alpha) - alpha[2]))/den
    cov_theta12 = - ((alpha[0]*alpha[1])/den)
    cov_theta13 = - ((alpha[0]*alpha[2])/den)
    cov_theta23 = - ((alpha[1]*alpha[2])/den)
    
    sigmas = np.array([[var_theta1, cov_theta12, cov_theta13], 
                       [cov_theta12, var_theta2, cov_theta23], 
                       [cov_theta13, cov_theta23, var_theta3]])
    
    mus = np.array([0, 0, 0])
    
    x, y, z = 0.3, 0.4, 0.3
    samples = np.zeros((n, 3))
    
    for i in range(n):
    
        x_star, y_star, z_star = np.array([x, y, z]) + np.random.multivariate_normal(mus, sigmas )
        
        g_x = st.dirichlet.pdf(np.array([x,y,z]), alpha)
        
        g_x_star = st.dirichlet.pdf(np.array([x_star,y_star,z_star]), alpha)
        
        a = np.min(1.0, g_x_star/g_x)
    
        if np.random.rand() < a:
            x, y, z = x_star, y_star, z_star
            print(np.sum(np.array([x,y,z])))
        samples[i] = np.array([x, y, z])
    

    return samples
n = 1000
x1 = np.array([3,4,5])
y1 = np.array([4,4,4])
samples = thetamcmc(n, x1,y1)
