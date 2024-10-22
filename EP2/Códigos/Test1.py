from random import random, uniform
from math import cos, exp,sqrt
from statistics import variance
a = 0.541147330
b = 0.42490023810

def f(x):
    return exp(-a*x)*cos(b*x)
n = 0
gamma = 0
err = 1
list_f = []
while err > 0.0005:
    n += 1
    x1 = random()
    list_f.append(f(x1))
    gamma = gamma + f(x1)
    if n == 1:
        err = 1
    else:
        # l = len(list_f)
        # var = 0
        # for i in range(l):
        #     var = var + 1/(n-1)*(list_f[i]-integral/n)**2
        var = variance(list_f)
        err = sqrt(var/n)

gamma_f = gamma/n
print(gamma_f)