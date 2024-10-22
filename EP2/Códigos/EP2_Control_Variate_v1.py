## Control Variate
from math import exp, cos, sqrt
from random import random, uniform, seed
from time import time
from scipy.stats import t
def Print_info(media, n, err, t):
    print("Media: ", media)
    print("O numero de iterações: ", n)
    print("O erro: ", err*100, "%")
    print("Tempo de simulação: ", t)
  
a = 0.541147330
b = 0.42490023810
Integral_exp = 1/a*(1-exp(-a))
Integral_1 = 1-a/2 #+(a**2-b**2)/3

def f(x):
    return exp(-a*x)*cos(b*x)
def g(x):
    return exp(-a*x)
def h(x):
    return 1-a*x#+(a**2-b**2)*x**2 

seed(81)
def Control_Variate(f,g, Integral_known):
    n = 0
    soma_f = 0
    soma_g = 0
    soma_quad_f = 0
    soma_quad_g = 0
    soma_fg = 0
    list_f = []
    list_g = []
    err = 1
    t1 = time()
    while err > 0.0005:
        n += 1
        x = random()
        f1 = f(x)
        g1 = g(x)
        soma_f = soma_f + f1
        soma_g = soma_g + g1
        soma_quad_f = soma_quad_f + f1**2
        soma_quad_g = soma_quad_g + g1**2
        soma_fg = soma_fg + f1*g1
        list_f.append(f1)
        list_g.append(g1)
        if n == 1:
            err = 1
        else:
            Cov = (soma_fg - soma_f*soma_g/n)/(n-1)
            var_f = (soma_quad_f - soma_f**2/n)/(n-1)
            var_g = (soma_quad_g - soma_g**2/n)/(n-1)
            var = var_f + var_g -2*Cov
            err = -t.ppf(0.005,n)*sqrt(var/n)
    t2 = time()
    media = (soma_f-soma_g)/n + Integral_known
    return media, n, err, t2-t1

media1, n1, err1, t1 = Control_Variate(f,g, Integral_exp)
media2, n2, err2, t2 = Control_Variate(f,h, Integral_1)
print("Control Variate Monte Carlo Method:")
print("phi(x) = exp(-ax) :")
Print_info(media1, n1, err1, t1)
print("\n")
print("phi(x) = 1-ax :")
Print_info(media2, n2, err2, t2)