from scipy.special import gamma
import numpy as np
import statistics
import scipy.stats as st
from time import time

def x():

    vetor_x = np.zeros((3,))
    
    x1 = int(input("Escolha x1: "))
    x2 = int(input("Escolha x2: "))
    x3 = int(input("Escolha x3: "))

    vetor_x[0] = x1
    vetor_x[1] = x2
    vetor_x[2] = x3
    
    return vetor_x

def y():
    
    vetor_y = np.zeros((3,))
    
    y1 = int(input("Escolha y1: "))
    y2 = int(input("Escolha y2: "))
    y3 = int(input("Escolha y3: "))

    vetor_y[0] = y1
    vetor_y[1] = y2
    vetor_y[2] = y3
    
    return vetor_y


def theta(n, x, y):
    
    params = x + y

    vetor_theta = np.random.dirichlet(params, n)

    return vetor_theta

def f(theta, x, y):
    
    params = x + y
    
    beta = np.prod(gamma(params))/ gamma(np.sum(params))
    
    resul = (1/beta)*np.prod(theta**(x + y - 1), axis = 1)
    
    return resul

def hitu(v1, v2, n, theta, x, y):
    
    est_w = 1/n*len(np.argwhere((f(theta, x, y) <= v2) & (f(theta, x, y) >= v1)))
    
    return est_w


def g(theta, params):
    
    if theta[0] < 0 or theta[1] < 0 or theta[2] < 0:
        return -1

    # beta = np.prod(gamma(params))/ gamma(np.sum(params))
    
    # resul = (1/beta)*np.prod(theta**(params - 1))
    resul = np.prod(theta**(params - 1))
    return resul

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
    burn_in = 200
    # mean = np.array([0,0,0])
    sample = np.zeros((n+burn_in,3))
    sample[0] = theta_old
    # i = 0
    # while i < burn_in+n:
    for i in range(burn_in+n):
        mean = theta_old
        sample[i] = theta_old
        x_star, y_star, z = np.random.multivariate_normal(mean, cov)
        z_star = 1 - (x_star + y_star)
        theta_star = np.array([x_star, y_star, z_star])
        alpha = np.min([g(theta_star,xy)/g(theta_old,xy),1])

        u = np.random.uniform()
        if u <= alpha:
            theta_old = theta_star
            # sample[i] = theta_old
            # i = i+1
        # i = i+1   
    sample = sample[burn_in:]
    
    t2 = time()
    return sample
    
def umcmc():
    t1 = time()
    vetor_x = x() # chama o vetor x para o usuário escolher
    vetor_y = y() # chama o vetor y para o usuário escolher
    
    k_inicial = 2000 # ARRUMAR ISSO
    
    vk = float(input("Escolha v: ")) # pede pro usuário escolher o corte v (até onde a integral será calculada)
    
    n_inicial = 100000 
    
    # calcula estimativa
    
    vetor_theta = thetamcmc(n_inicial, vetor_x, vetor_y) # calcula o vetor theta
        
    
    ff = f(vetor_theta, vetor_x, vetor_y) # calcula a função densidade de probabilidade posteriori de Dirichlet 
        
    # aqui começa a se calcular a lista de cortes
    cort = np.zeros((1,)) # primeiro corte v = 0
    cortes = statistics.quantiles(ff, n = k_inicial) # segundo corte até o k-ésimo corte (sem os extremos, sem o primeiro e último corte)
    cortes = np.concatenate((cort, cortes), axis = 0) # junta o primeiro com o resto
    maxi = np.array([np.max(ff)]) # (k+1)-ésimo corte, ou seja, último corte, v  = sup(f)
    cortes = np.concatenate((cortes, maxi), axis = 0) # junta todos os cortes
    # a lista tem k + 1 elementos, pontos de corte, resultando em k bins
    # de modo que a área em cada bin é 1/k
            
    v = cortes[cortes <= vk] # considera apenas os elementos até o corte escolhido pelo usuário
    # calcula a área: soma da área de todos os bins da lista
    # como a área em cada bin é 1/k basta multiplicar 1/k pelo número total de bins (comprimento da lista - 1 no caso)
    area = 1/k_inicial*(len(v) - 1)
    if vk not in v and len(v) < k_inicial + 1: # se o corte escolhido não esta na lista e o comprimento da lista diminuiu
        # calcula a área do ultimo valor da lista até o corte escolhido e soma na área
        area += hitu(v[-1], vk, n_inicial, vetor_theta, vetor_x, vetor_y)
    
    n_novo = int((area*(1 - area))*(2.575/0.0005)**2)
    k_novo = int(4*(2*n_novo**2/2.575)**(1/5))
    if n_novo < n_inicial:
        n_novo = n_inicial
    if k_novo < k_inicial:
        k_novo = k_inicial
    # após ter achado n
    
    vetor_theta = thetamcmc(n_novo, vetor_x, vetor_y) # calcula o vetor theta com n certo
    
    ff = f(vetor_theta, vetor_x, vetor_y) # calcula a função densidade de probabilidade posteriori de Dirichlet

    # calcula de novo os cortes, do mesmo modo que anteriormente
    cort = np.zeros((1,)) # primeiro corte v = 0
    cortes = statistics.quantiles(ff, n = k_novo) # segundo corte até o k-ésimo corte (sem os extremos, sem o primeiro e último corte)
    cortes = np.concatenate((cort, cortes), axis = 0) # junta o primeiro com o resto
    maxi = np.array([np.max(ff)]) # (k+1)-ésimo corte, ou seja, último corte, v  = sup(f)
    cortes = np.concatenate((cortes, maxi), axis = 0) # junta todos os cortes
    # a lista tem k + 1 elementos, pontos de corte, resultando em k bins
    # de modo que a área em cada bin é 1/k

    # calcula a area (estimativa da integral) do mesmo modo
    v = cortes[cortes <= vk] # considera apenas os elementos até o corte escolhido pelo usuário
    # calcula a área: soma da área de todos os bins da lista
    # como a área em cada bin é 1/k basta multiplicar 1/k pelo número total de bins (comprimento da lista - 1 no caso)
    area = 1/k_novo*(len(v) - 1)
    if vk not in v and len(v) < k_novo + 1: # se o corte escolhido não esta na lista e o comprimento da lista diminuiu
        # calcula a área do ultimo valor da lista até o corte escolhido e soma na área
        area += hitu(v[-1], vk, n_inicial, vetor_theta, vetor_x, vetor_y) 
    t2=time()
    # printa o valor da área
    print(n_novo)
    print(area)
    print(t2-t1)
umcmc()