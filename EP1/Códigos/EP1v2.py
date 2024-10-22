## Importa funções utilizadas no EP
from random import uniform as random
# from random import random
from math import sqrt, pi 
from statistics import stdev, mean
import matplotlib.pyplot as plt
from scipy.stats import t

## Procura n que
n_1 = 10**5
n_r = 1
err_p = 1
# list_A =[] ## Lista das áreas obtidas através do método estocástico apresentado em aula
## Lista dos erros das áreas em relação ao resultado analítico
list_p = []
z_alpha = 3.9
cont = 0
while err_p > 0.05:
    n = n_1
    list_sample = []
    # for j in range(n_r):
    #     A = 0 ## Área que será calculada inicia com o valor zero
    ##
    p = 0

    for k in range(n):
        # x = 2*random()-1
        # y = 2*random()-1
        x = random(-1,1)
        y = random(-1,1)
        if sqrt((x**2 + y**2)) <= 1 :
            p = p+1/n
    A = 4*p
    # err_A = 100*(A-pi)/(pi)
    list_p.append(p)
    # list_sample.append(A)
    ##
    # mean_p = mean(list_p)
    # stdev_p = stdev(list_p)
    err_p = 100*(z_alpha*sqrt(p*(1-p)/n))
    n_1 = round((z_alpha/(0.0005*p))**2*p*(1-p))
    # list_A.append(list_sample)
    cont+=1

print(cont)
print(4*p)
print(err_p)
print(n)