#### Importa funções utilizadas no EP
from random import uniform as random
# from random import random
from math import sqrt, pi 
from statistics import stdev, mean
import matplotlib.pyplot as plt
from scipy.stats import t

#### Parâmetros iniciais
n_1 = 10**5 ## Chute inicial para n
err_p = 1 ## Valor do erro percentual, colocou-se 1 para entrar no laço
z_alpha = 3.3 ## Valor da variável z para alpha = 0.1%, esse valor muda para alpha diferentes
cont = 0 ## Contador que mostra quantas iterações foram feitas

#### Listas das variáveis
list_n = [] ## Lista dos n
list_p = [] ## Lista dos valores obtidos para p_n para cada n
list_err = [] ## Lista dos erros percentuais obtidos para cada n
list_pi = [] ## Lista dos pi's estimados
#### Algoritmo
while err_p > 0.05: ## Condiciona a saída do laço somente se o erro percentual for menor que 0.05%
    n = n_1 ## Substitui o antigo valor de n para um novo valor de n
    p = 0 ## Inicializa a proporção p_n com zero a cada rodada desse laço
    for k in range(n): ## Laço para para gerar os pontos aleatoriamente
        y = random(-1,1) ## Coordenada y do ponto x_i
        z = random(-1,1) ## Coordenada z do ponto x_i
        if sqrt((y**2 + z**2)) <= 1 : ## Faz o papel da função T(x) se o ponto estiver dentro da circunferência ele atribui 1 e se não
            T = 1
        else:
            T = 0
        p = p + T/n ## Faz o papel do termo: 1/n vezes somatoria de T de 1 até n

    pi_estimado = 4*p ## Calcula o pi_estimado nessa rodada 
    err_p = 100*(z_alpha*sqrt(p*(1-p)/n))/p ## Calcula o erro percentual 
    n_1 = round((z_alpha/(0.0005*p))**2*p*(1-p)) ## Recalcula o n
    cont+=1

    list_p.append(p) ## Indexa o valor de p_n dentro da lista 
    list_err.append(err_p) ## Indexa o valor do erro percentual nessa rodada
    list_pi.append(pi_estimado) ## Indexa o valor do pi estimado nessa rodada
    list_n.append(n) ## Indexa o valor de n utilizado nessa rodada

#### Imprime as variáveis
print('Número de iterações: ', cont, '\n')
print('Lista das proporções p_n: ', list_p, '\n')
print('Lista dos n: ', list_n, '\n')
print('Lista dos erros percentuais: ', list_err, 'n')
print('Lista dos pi estimados: ', list_pi, '\n')
print('n estimado final: ',list_n[-1], '\n')
print('pi estimado final: ', list_pi[-1], '\n')
