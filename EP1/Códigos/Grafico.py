## Importa funções utilizadas no EP
from random import uniform as random
# from random import random
from math import sqrt, pi 
from statistics import stdev, mean
import matplotlib.pyplot as plt

# #### Funções criadas
## Primeira função: Utilizada para repetir n vezes cada elemento da lista
## Exemplo: lista = [1, 2, 3] e n = 2 
##          lista2 = repmat(lista, n) = [[1, 1], [2, 2], [3, 3]]
def repmat(list_1, n):
    list_2 = []
    l= len(list_1)
    for j in range(l):
        list_aux = []
        for i in range(n):
            list_aux.append(list_1[j])

        list_2.append(list_aux)
    return list_2


### Parâmetros de entrada

## Lista dos n's utilizados para criar um gráfico 
list_n = [1*10**5,4*10**5, 8*10**5, 12*10**5, 16*10**5] 

## Número de vezes que será repetido o "experimento" com um certo n
n_r = 10

#### Variáveis calculadas a partir dos parâmetros de entrada

## Tamanho da lista dos n's
l_n = len(list_n) 

## Lista utilizada para plotar gráfico
list_plot = repmat(list_n, n_r) 

#### Lista de variáveis calculadas nos "experimentos"
 
list_pi =[] ## Lista das áreas obtidas através do método estocástico apresentado em aula
list_err = [] ## Lista dos erros das áreas em relação ao resultado analítico
list_p = []
for i in range(l_n):
    n = list_n[i]
    list_sample = []
    for j in range(n_r):
        pi_estimado = 0 ## Valor de pi_estimado inicializa com zero a cada rodada
        p = 0 ## Valor da proporção p_n inicializa com zero a cada rodada
        for k in range(n):
            # x = 2*random()-1
            # y = 2*random()-1
            x = random(-1,1) ## Chuta valor da coordenada x do ponto
            y = random(-1,1)
            if sqrt((x**2 + y**2)) <= 1 :
                p = p+1/n
        pi_estimado = 4*p

        list_p.append(p)
        list_sample.append(pi_estimado)
    list_pi.append(list_sample)


plt.figure(1)
plt.plot(list_plot, list_pi, 'ro')
plt.show()