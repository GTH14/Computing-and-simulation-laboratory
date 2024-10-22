from math import exp, cos, sqrt
from random import random, uniform

a = 0.541147330
b = 0.42490023810

def f(x):
    return exp(-a*x)*cos(b*x)

def h(x,y):
    if y <= f(x):
        return 1
    else:
        return 0

n = 0
gamma_tot = 0
gamma = 0
err = 1
while err>0.0005:
    n += 1
    x = random()
    y = random()
    gamma_tot = gamma_tot + h(x,y)
    gamma = gamma_tot/n
    if n == 1:
        err = 1
    else:
        err = sqrt(gamma*(1-gamma)/n)
print(n)
print(gamma)