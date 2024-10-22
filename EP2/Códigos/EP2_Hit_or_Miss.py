## Hit or Miss Monte Carlo Method

# Importa as funções utilizadas nesse código
from math import exp, cos, sqrt
from random import random, uniform, seed
from time import time

# Define os parâmetros de entrada
a = 0.541147330 # 0.RG
b = 0.42490023810 #0.CPF
seed(10) # Define uma seed para os resultados obtidos pelo aluno 
#sejam os mesmos obtidos pelo monitor

# Define as funções
def Print_info(media, n, err, t): # Função para printar os resultados obtidos no método
    print("Media: ", media)
    print("O numero de iterações: ", n)
    print("O erro: ", err*100, "%")
    print("Tempo de simulação: ", t)

def f(x):# Função que se quer integrar
    return exp(-a*x)*cos(b*x)

def T(x,y): # Função que testa se y<=f(x)
    if y <= f(x):
        return 1
    else:
        return 0

# Define o método Hit or Miss para estimar o valor da integral
def Hit_Miss_MC():
    n = 0 # Inicializa a variável n que representa o número de iterações necessária
    #para obter um erro menor que 0.05%
    soma = 0 # Inicializa o somatório de T(x_i,y_i) de i = 1 até n
    err = 1 # Inicializa a variável do erro relativo
    t1 = time() # Utilizado para calcular o tempo para rodar a função
    z_alpha = 3.3 # Valor da variável z para alpha = 0.1%, esse valor muda para alpha diferente
    while err>0.0005: # Só sairá do loop se o erro relativo for menor que 0.0005 ou 0.05%
        n +=1
        # for i in range(n):
        x = random() # Atribui o valor a variável aleatória x_i
        y = random() # Atribui o valor a variável aleatória y_i
        soma = soma + T(x,y) # Adiciona termo ao somatório de T(x_i, y_i)
        if n == 1 or soma == 0 or soma == n: # Condição para que não saia do loop, pois
        # caso uma dessas condições for satisfeita o erro relativo vai ser igual a 0
            err = 1
        else:
            err = z_alpha*sqrt(soma/n*(1-soma/n)/n) # Calcula o erro relativo do método para dado n
            # n1 = round((3/(0.0005*(soma/n)))**2*soma/n*(1-soma/n))
    t2 = time() 
    media = soma/n # Calcula o estimador para a integral de f(x)
    return media, n, err, t2-t1
media, n, err, t = Hit_Miss_MC()
print("Hit or Miss Carlo Method:")
Print_info(media, n, err, t)