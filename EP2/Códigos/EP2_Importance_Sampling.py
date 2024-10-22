## Importance sampling

# Importa as funções utilizadas nesse código
from math import exp, cos, sqrt
from scipy.stats import beta, t
from random import betavariate, seed
from time import time
import chaospy as cp
# Define os parâmetros de entrada
a = 0.541147330 # 0.RG
b = 0.42490023810 #0.CPF
alfa = 0.9 # Parâmetro alfa para a função Beta
betha = 1.05 # Parâmetro beta para a função Beta
seed(21) # Define uma seed para os resultados obtidos pelo aluno 
#sejam os mesmos obtidos pelo monitor

# Define as funções
def Print_info(media, n, err, t): # Função para printar os resultados obtidos no método
    print("Media: ", media)
    print("O numero de iterações: ", n)
    print("O erro: ", err*100, "%")
    print("Tempo de simulação: ", t)

def f(x): # Função que se quer integrar
    return exp(-a*x)*cos(b*x)

def g(x): # Função distribuição de probabilidade da variável aleatória
    return beta.pdf(x,alfa, betha, loc=0, scale=1)

# Define o método Importance Sampling para estimar o valor da integral
def Importance_Sampling():
    n = 0 # Inicializa a variável n que representa o número de iterações necessária
    #para obter um erro menor que 0.05%
    soma = 0 # Inicializa o somatório de f(x_i)/g(x_i) de i = 1 até n
    soma_quad = 0 # Inicializa o somatório de (f(x_i)/g(x_i))^2 de i = 1 até n
    # list_r = [] 
    err = 1 # Inicializa a variável do erro relativo
    t1 = time() # Utilizado para calcular o tempo para rodar a função
    while err > 0.0005: # Só sairá do loop se o erro relativo for menor que 0.0005 ou 0.05%
        n += 1
        x = cp.create_halton_samples(1,1,-1+n) # Atribui o valor a variável aleatória x_i
        x1 = beta.ppf(x, alfa, betha)
        f_1 = f(x1) # Calcula f(x_i)
        g_1 = g(x1) # Calcula g(x_i)
        # list_r.append(f_1/g_1)
        soma = soma + f_1/g_1 # Adiciona termo ao somatório de f(x_i)/g(x_i)
        soma_quad = soma_quad + (f_1/g_1)**2 # Adiciona termo ao somatório de (f(x_i)/g(x_i))^2
        if n == 1: # Condição para que não dê erro na primeira roda pois há um divisão por n-1
        # e quando n = 1 haverá divisão por zero, ou seja, dará um erro
            err = 1
        else:
            var = (soma_quad - soma**2/n)/(n-1) # Calcula a variância do método
            # err = -t.ppf(0.005,n-1)*sqrt(var/n) # Calcula o erro desse método para dado n
            err = sqrt(var/n)
    t2 = time()
    media = soma/n # Calcula o estimador para a integral de f(x)
    return media, n, err, t2-t1 # Retorna a estimativa, o n, o erro e o tempo para rodar a rotina

media, n, err, t = Importance_Sampling()
print("Importance Sampling Monte Carlo Method:")
Print_info(media, n, err, t)