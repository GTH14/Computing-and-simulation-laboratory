## Control Variate

# Importa as funções utilizadas nesse código
from math import exp, cos, sqrt
from random import random, seed
from time import time
from scipy.stats import t
import chaospy as cp

# Define os parâmetros de entrada
a = 0.541147330 # 0.RG
b = 0.42490023810 #0.CPF
Integral_exp = 1/a*(1-exp(-a)) # Valor da integral de 0 a 1 de exp(-ax)
Integral_1 = 1-a/2  # Valor da integral de 0 a 1 de 1-ax, Polinômio de Taylor de grau 1
seed(81) # Define uma seed para os resultados obtidos pelo aluno 
#sejam os mesmos obtidos pelo monitor
# Define as funções

def Print_info(media, n, err, t): # Função para printar os resultados obtidos no método
    print("Media: ", media)
    print("O numero de iterações: ", n)
    print("O erro: ", err*100, "%")
    print("Tempo de simulação: ", t)

def f(x): # Função que se quer integrar
    return exp(-a*x)*cos(b*x)

def phi1(x): # Função utilizada como variável de controle 
    return exp(-a*x)

def phi2(x): # Função utilizada como variável de controle 
    return 1-a*x

# Define o método Control Variate para estimar o valor da integral
def Control_Variate(f,phi, Integral_known):
    n = 0 # Inicializa a variável n que representa o número de iterações necessária
    #para obter um erro menor que 0.05%
    soma_f = 0 # Inicializa o somatório de f(x_i) de i = 1 até n
    soma_phi = 0 # Inicializa o somatório de g(x_i) de i = 1 até n
    soma_quad_f = 0 # Inicializa o somatório de f(x_i)^2 de i = 1 até n
    soma_quad_phi = 0 # Inicializa o somatório de g(x_i)^2 de i = 1 até n
    soma_fphi = 0 # Inicializa o somatório de f(x_i)*g(x_i) de i = 1 até n
    err = 1 # Inicializa a variável do erro relativo
    t1 = time() # Utilizado para calcular o tempo para rodar a função
    while err > 0.0005: # Só sairá do loop se o erro relativo for menor que 0.0005 ou 0.05%
        n += 1
        x = cp.create_halton_samples(1,1, -1+n) # Atribui o valor a variável aleatória x_i
        # x = random()
        f1 = f(x) # Calcula f(x_i)
        phi1 = phi(x) # Calcula phi(x_i)
        soma_f = soma_f + f1 # Adiciona termo ao somatório de f(x_i)
        soma_phi = soma_phi + phi1 # Adiciona termo ao somatório de phi(x_i)
        soma_quad_f = soma_quad_f + f1**2 # Adiciona termo ao somatório de f(x_i)^2
        soma_quad_phi = soma_quad_phi + phi1**2 # Adiciona termo ao somatório de phi(x_i)^2
        soma_fphi = soma_fphi + f1*phi1 # Adiciona termo ao somatório de f(x_i)*phi(x_i)
        if n == 1:# Condição para que não dê erro na primeira roda pois há um divisão por n-1
        # e quando n = 1 haverá divisão por zero, ou seja, dará um erro
            err = 1
        else:
            Cov = (soma_fphi - soma_f*soma_phi/n)/(n-1) # Calcula a covariância das funções
            # f(x) e phi(x)
            var_f = (soma_quad_f - soma_f**2/n)/(n-1) # Calcula a variância de f(x)
            var_phi = (soma_quad_phi - soma_phi**2/n)/(n-1) # Calcula a variância de phi(x)
            var = var_f + var_phi -2*Cov # Calcula a variância desse método
            # err = -t.ppf(0.005,n-1)*sqrt(var/n) # Calcula o erro desse método para dado n
            err = 2.575*sqrt(var/n)
    t2 = time()
    media = (soma_f-soma_phi)/n + Integral_known # Calcula o estimador para o valor da integral de f(x)
    return media, n, err, t2-t1 # Retorna a estimativa, o n, o erro e o tempo para rodar a rotina

media1, n1, err1, t1 = Control_Variate(f,phi1, Integral_exp) # Utiliza phi(x) = exp(-a*x)
media2, n2, err2, t2 = Control_Variate(f,phi2, Integral_1) # Utiliza phi(x) = 1 - a*x
print("Control Variate Monte Carlo Method:")
print("phi(x) = exp(-a*x) :")
Print_info(media1, n1, err1, t1)
print("\n")
print("phi(x) = 1-a*x :")
Print_info(media2, n2, err2, t2)