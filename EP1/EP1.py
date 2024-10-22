## Importa funções utilizadas no EP
from random import uniform as random
# from random import random
from math import sqrt, pi 
from statistics import stdev, mean
import matplotlib.pyplot as plt

# #### Funções criadas
## Primeira função: Utilizada para repetir n vezes cada elemento da lista
## Exemplo: lista = [1, 2, 3] e n = 2 
##          lista2 = repmat(lista, n) = [1, 1, 2, 2, 3, 3]
def repmat(list_1, n):
    list_2 = []
    l= len(list_1)
    for j in range(l):
        list_aux = []
        for i in range(n):
            list_aux.append(list_1[j])

        list_2.append(list_aux)
    return list_2

## Segunda função: calcula as médias de intervalos com n numeros da lista. É feito de forma sequencial, ou seja,
## pega-se os primeiros n números da lista e calcula-se a média, repete-se esse procedimento para os próximos n
## números, e depois para os próximos n números até chegar o último intervalo com n números da lista. 
## Exemplo: lista = [1, 2, 3, 4, 5, 6, 7, 8, 9] e n = 3
##          lista = calc_media(lista, n) = [2, 5, 8]
## Obs.: o tamanho da lista deve ser um múltiplo de n
def calc_media(list_1):
    list_2 = []
    l = len(list_1)
    n = len(list_1[0])
    for i in range(l):
        list_2.append(mean(list_1[i]))
    return list_2

## Terceira função: Ideia parecida com a função anterior, contudo calcula-se o desvio padrão dos intervalos com n números
## Exemplo: lista = [1, 1, 1, 2, 3, 4, 5, 5, 6] e n = 3
##          lista2 = calc_desv(lista, n) = [0.0, 1.0, 0.5773502691896257]
def calc_desv(list_1, n):
    list_2 = []
    l = len(list_1)
    cont = 0
    for i in range(l):
        samples = list_1[i]
        list_2.append(stdev(samples))
    return list_2

### Parâmetros de entrada

## Lista dos n's utilizados para criar um gráfico para análise
## e determinação do n que se adeque a precisão requisitada
list_n = [2*10**5,2*10**6] 

## Número de vezes que será repetido o "experimento" com um certo n
n_r = 10

## Raio do círculo, no caso é unitário
r = 1 

#### Variáveis calculadas a partir dos parâmetros de entrada

## Tamanho da lista dos n's
l_n = len(list_n) 

## Lista utilizada para plotar gráficos
list_plot = repmat(list_n, n_r) 

#### Lista de variáveis calculadas nos "experimentos"
 
list_A =[] ## Lista das áreas obtidas através do método estocástico apresentado em aula
list_err = [] ## Lista dos erros das áreas em relação ao resultado analítico
list_p = []
for i in range(l_n):
    n = list_n[i]
    list_sample = []
    for j in range(n_r):
        A = 0 ## Área que será calculada inicia com o valor zero
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
        list_sample.append(A)
    list_A.append(list_sample)
        # list_err.append(err_A)

list_media_A = calc_media(list_A)
# list_media_err = calc_media(list_err, n_r)
list_desvpad_A = calc_desv(list_A, n_r)
# list_desvpad_err = calc_desv(list_err, n_r)


plt.figure(1)
plt.plot(list_plot, list_A, 'ro')

plt.figure(2)
plt.plot(list_n, list_media_A, 'ro')

plt.figure(3)
plt.plot(list_n, list_desvpad_A, 'ro')

# plt.figure(4)
# plt.plot(list_plot, list_err, 'ro')

print(list_A)
plt.show()
        