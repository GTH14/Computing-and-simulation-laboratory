import numpy as np
import statistics

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
    
    resul = np.prod(theta**(x + y - 1), axis = 1)
    
    return resul

def hitu(v1, v2, n, theta, x, y):
    
    est_w = 1/n*len(np.argwhere((f(theta, x, y) <= v2) & (f(theta, x, y) >= v1)))
    
    return est_w

def u(k):
    
    vetor_x = x()
    vetor_y = y()
    
    var = (1/k)*(1 - 1/k)
    z, erro = 2.575, 0.0005

    n = int(((z/erro)**2)*var)
    print(n)
    vetor_theta = theta(n, vetor_x, vetor_y)
    
    
    ff = f(vetor_theta, vetor_x, vetor_y)
    
    cortes = np.array(statistics.quantiles(ff, n = k, method = 'inclusive'))
    
    def U(vk):
        v = cortes[cortes <= vk]
        if vk not in v:
            v = np.concatenate((v, np.array([vk])), axis = 0)
        
        soma = []
        for i in range(1, len(v)):
            a = hitu(v[i-1], v[i], n, vetor_theta, vetor_x, vetor_y)
            soma.append(a)
            
        
        integral = np.sum(np.array(soma))
        
        return integral
    return U

U1 = u(100000)
print(U1(0.00000000000001))