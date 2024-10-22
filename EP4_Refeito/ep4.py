from scipy.special import gamma
import numpy as np
import statistics


# Vetor X
def x():

    vetor_x = np.zeros((3,)) # começa zerado

    # escolhe os valores do vetor x
    x1 = int(input("Escolha x1: "))
    x2 = int(input("Escolha x2: "))
    x3 = int(input("Escolha x3: "))

    # aplica os valores escolhidos ao vetor
    vetor_x[0] = x1
    vetor_x[1] = x2
    vetor_x[2] = x3

    # retorna o vetor x
    return vetor_x

# Vetor Y
def y():
    
    vetor_y = np.zeros((3,)) # começa zerado

    # escolhe os valores do vetor y
    y1 = int(input("Escolha y1: "))
    y2 = int(input("Escolha y2: "))
    y3 = int(input("Escolha y3: "))

    # aplica os valores escolhidos ao vetor
    vetor_y[0] = y1
    vetor_y[1] = y2
    vetor_y[2] = y3

    # retorna o vetor y
    return vetor_y


# Vetor Theta
def theta(n, x, y):

    # a soma dos vetores são os parametros para a dirichlet
    params = x + y

    # gera n triplas com distribuição dirichlet
    vetor_theta = np.random.dirichlet(params, n)

    # retorna o vetor theta
    return vetor_theta

# Função densidade de probabilidade posteriori de Dirichlet 
def f(theta, x, y):

    # a soma dos vetores são os parametros para a função beta
    params = x + y

    # a função beta
    beta = np.prod(gamma(params))/ gamma(np.sum(params))

    # a função de densidade em si
    # devolve n valores
    resul = (1/beta)*np.prod(theta**(x + y - 1), axis = 1)

    # retorna os valores da função
    return resul

# Função "Hit-or-Miss"
def hitu(v1, v2, n, theta, x, y):

    # checa a quantidade de valores entre v1 e v2
    # divide pelo número total de pontos
    # se v1 e v2 são pontos de corte consecutivos, est_w será igual a 1/k
    est_w = 1/n*len(np.argwhere((f(theta, x, y) <= v2) & (f(theta, x, y) >= v1)))

    # retorna o valor encontrado
    return est_w

def u():
    vetor_x = x() # chama o vetor x para o usuário escolher
    vetor_y = y() # chama o vetor y para o usuário escolher
    
    k_inicial = 2000 # para a estimativa piloto
    
    vk = float(input("Escolha v: ")) # pede pro usuário escolher o corte v (até onde a integral será calculada)
    
    n_inicial = 100000 # para a estimativa piloto
    
    # calcula estimativa
    
    vetor_theta = theta(n_inicial, vetor_x, vetor_y) # calcula o vetor theta
        
    
    ff = f(vetor_theta, vetor_x, vetor_y) # calcula a função densidade de probabilidade posteriori de Dirichlet 
        
    # aqui começa a se calcular a lista de cortes
    cort = np.zeros((1,)) # primeiro corte v = 0
    cortes = statistics.quantiles(ff, n = k_inicial) # segundo corte até o k-ésimo corte (sem os extremos, sem o primeiro e último corte)
    cortes = np.concatenate((cort, cortes), axis = 0) # junta o primeiro com o resto
    maxi = np.array([np.max(ff)]) # (k+1)-ésimo corte, ou seja, último corte, v  = sup(f)
    cortes = np.concatenate((cortes, maxi), axis = 0) # junta todos os cortes
    # a lista tem k + 1 elementos, pontos de corte, resultando em k bins
    # de modo que a área em cada bin é 1/k
            
    v = cortes[cortes <= vk] # considera apenas os elementos até o corte escolhido pelo usuário
    # calcula a área: soma da área de todos os bins da lista
    # como a área em cada bin é 1/k basta multiplicar 1/k pelo número total de bins (comprimento da lista - 1 no caso)
    area = 1/k_inicial*(len(v) - 1)
    if vk not in v and len(v) < k_inicial + 1: # se o corte escolhido não esta na lista e o comprimento da lista diminuiu
        # calcula a área do ultimo valor da lista até o corte escolhido e soma na área
        area += hitu(v[-1], vk, n_inicial, vetor_theta, vetor_x, vetor_y)
    
    n_novo = int((area*(1 - area))*(2.575/0.0005)**2) # calcula novo n (explicação no relatório)
    k_novo = int(4*(2*n_novo**2/2.575)**(1/5)) # calcula novo k (explicação no relatório)
    if n_novo < n_inicial: # se o novo n for menor que o antigo, mantém o antigo
        n_novo = n_inicial
    if k_novo < k_inicial: # se o novo k for menor que o antigo, mantém o antigo
        k_novo = k_inicial

    # após ter achado n e k
    
    vetor_theta = theta(n_novo, vetor_x, vetor_y) # calcula o vetor theta com n certo
    
    ff = f(vetor_theta, vetor_x, vetor_y) # calcula a função densidade de probabilidade posteriori de Dirichlet

    # calcula de novo os cortes, com k certo, do mesmo modo que anteriormente
    cort = np.zeros((1,)) # primeiro corte v = 0
    cortes = statistics.quantiles(ff, n = k_novo) # segundo corte até o k-ésimo corte (sem os extremos, sem o primeiro e último corte)
    cortes = np.concatenate((cort, cortes), axis = 0) # junta o primeiro com o resto
    maxi = np.array([np.max(ff)]) # (k+1)-ésimo corte, ou seja, último corte, v  = sup(f)
    cortes = np.concatenate((cortes, maxi), axis = 0) # junta todos os cortes
    # a lista tem k + 1 elementos, pontos de corte, resultando em k bins
    # de modo que a área em cada bin é 1/k

    # calcula a area (estimativa da integral) do mesmo modo
    v = cortes[cortes <= vk] # considera apenas os elementos até o corte escolhido pelo usuário
    # calcula a área: soma da área de todos os bins da lista
    # como a área em cada bin é 1/k basta multiplicar 1/k pelo número total de bins (comprimento da lista - 1 no caso)
    area = 1/k_novo*(len(v) - 1)
    if vk not in v and len(v) < k_novo + 1: # se o corte escolhido não esta na lista e o comprimento da lista diminuiu
        # calcula a área do ultimo valor da lista até o corte escolhido e soma na área
        area += hitu(v[-1], vk, n_inicial, vetor_theta, vetor_x, vetor_y) 
    # printa o valor da área
    print(area)
    
u()
