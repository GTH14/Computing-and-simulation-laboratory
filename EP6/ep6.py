from scipy.special import gamma
import numpy as np
import statistics
import scipy.stats as st

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

# função que calcula o máximo na hipótese H
def s_estrela(n, vetx, vety):
    
    theta = np.zeros((n,3)) # espaço H, inicialmente vazio
    
    theta[:, 0] = np.random.rand(n) # gera theta1 uniformemente entre 0 e 1
    theta[:, 2] = (1 - np.sqrt(theta[:, 0]))**2 # calcula theta3 a partir da fórmula
    theta[:, 1] = 1 - theta[:, 0] - theta[:, 2] # calula theta2 a partir da fórmula
    
    ff = f(theta, vetx, vety) # aplica na função posteriori de Dirichlet
    s_star = np.max(ff) # acha o máximo
    
    return s_star

# função que calcula W(s_star) = ev
def u(vetor_x, vetor_y):

    # caso a soma x + y tenha algum zero, retorna -1
    if vetor_x[2] == 0 and vetor_y[2] == 0:
        return -1

    # senao continua
    
    k_inicial = 2000 # para a estimativa piloto
    
    # valor maximo s_star
    vk = s_estrela(100000, vetor_x, vetor_y)
    
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
        area += hitu(v[-1], vk, n_novo, vetor_theta, vetor_x, vetor_y) 
    # printa o valor da área

    return area

# função que calcula o standarized e-value
def sev(ev):

    ev_barra = 1-ev # calcula ev_barra
    aux = st.chi2.ppf(ev_barra, 2, loc=0, scale=1) # calcula chi2^(-1)
    sev_barra = st.chi2.cdf(aux, 1, loc=0, scale=1) # calcula sev_barra a partir da fórmula dada pelo professor
    sev1 = 1-sev_barra # calcula sev
    return sev1

# função que printa as saídas
def saida(vx, vy):
    
    ev = u(vx,vy) # e-valor

    # se o e-valor é igual a -1 significa que caiu em alguma exceção, retorna Nada
    if ev == -1:
        print('Nada')

    # caso contrário    
    else:
        sevalue = sev(ev) # e-valor normalizado
    
        print('Standarized e-value:', sevalue) # printa o sev

        # aceita H se sev for maior ou igual a 0.05
        if sevalue >= 0.05:
            print('Aceitamos H para x =', vx, 'e y =', vy)
        # rejeita H caso contrário
        else:
            print('Rejeitamos H para x =', vx, 'e y =', vy)

# chama todos os vetores do artigo do professor, com y = 1 e y = 0
saida(np.array([1,17,2]), np.array([1,1,1]))
saida(np.array([1,17,2]), np.array([0,0,0]))
saida(np.array([1,16,3]), np.array([1,1,1]))
saida(np.array([1,16,3]), np.array([0,0,0]))
saida(np.array([1,15,4]), np.array([1,1,1]))
saida(np.array([1,15,4]), np.array([0,0,0]))
saida(np.array([1,14,5]), np.array([1,1,1]))
saida(np.array([1,14,5]), np.array([0,0,0]))
saida(np.array([1,13,6]), np.array([1,1,1]))
saida(np.array([1,13,6]), np.array([0,0,0]))
saida(np.array([1,12,7]), np.array([1,1,1]))
saida(np.array([1,12,7]), np.array([0,0,0]))
saida(np.array([1,11,8]), np.array([1,1,1]))
saida(np.array([1,11,8]), np.array([0,0,0]))
saida(np.array([1,10,9]), np.array([1,1,1]))
saida(np.array([1,10,9]), np.array([0,0,0]))
saida(np.array([1,9,10]), np.array([1,1,1]))
saida(np.array([1,9,10]), np.array([0,0,0]))
saida(np.array([1,8,11]), np.array([1,1,1]))
saida(np.array([1,8,11]), np.array([0,0,0]))
saida(np.array([1,7,12]), np.array([1,1,1]))
saida(np.array([1,7,12]), np.array([0,0,0]))
saida(np.array([1,6,13]), np.array([1,1,1]))
saida(np.array([1,6,13]), np.array([0,0,0]))
saida(np.array([1,5,14]), np.array([1,1,1]))
saida(np.array([1,5,14]), np.array([0,0,0]))
saida(np.array([1,4,15]), np.array([1,1,1]))
saida(np.array([1,4,15]), np.array([0,0,0]))
saida(np.array([1,3,16]), np.array([1,1,1]))
saida(np.array([1,3,16]), np.array([0,0,0]))
saida(np.array([1,2,17]), np.array([1,1,1]))
saida(np.array([1,2,17]), np.array([0,0,0]))
saida(np.array([1,1,18]), np.array([1,1,1]))
saida(np.array([1,1,18]), np.array([0,0,0]))
saida(np.array([5,15,0]), np.array([1,1,1]))
saida(np.array([5,15,0]), np.array([0,0,0]))
saida(np.array([5,14,1]), np.array([1,1,1]))
saida(np.array([5,14,1]), np.array([0,0,0]))
saida(np.array([5,13,2]), np.array([1,1,1]))
saida(np.array([5,13,2]), np.array([0,0,0]))
saida(np.array([5,12,3]), np.array([1,1,1]))
saida(np.array([5,12,3]), np.array([0,0,0]))
saida(np.array([5,11,4]), np.array([1,1,1]))
saida(np.array([5,11,4]), np.array([0,0,0]))
saida(np.array([5,10,5]), np.array([1,1,1]))
saida(np.array([5,10,5]), np.array([0,0,0]))
saida(np.array([5,9,6]), np.array([1,1,1]))
saida(np.array([5,9,6]), np.array([0,0,0]))
saida(np.array([5,8,7]), np.array([1,1,1]))
saida(np.array([5,8,7]), np.array([0,0,0]))
saida(np.array([5,7,8]), np.array([1,1,1]))
saida(np.array([5,7,8]), np.array([0,0,0]))
saida(np.array([5,6,9]), np.array([1,1,1]))
saida(np.array([5,6,9]), np.array([0,0,0]))
saida(np.array([5,5,10]), np.array([1,1,1]))
saida(np.array([5,5,10]), np.array([0,0,0]))
saida(np.array([9,11,0]), np.array([1,1,1]))
saida(np.array([9,11,0]), np.array([0,0,0]))
saida(np.array([9,10,1]), np.array([1,1,1]))
saida(np.array([9,10,1]), np.array([0,0,0]))
saida(np.array([9,9,2]), np.array([1,1,1]))
saida(np.array([9,9,2]), np.array([0,0,0]))
saida(np.array([9,8,3]), np.array([1,1,1]))
saida(np.array([9,8,3]), np.array([0,0,0]))
saida(np.array([9,7,4]), np.array([1,1,1]))
saida(np.array([9,7,4]), np.array([0,0,0]))
saida(np.array([9,6,5]), np.array([1,1,1]))
saida(np.array([9,6,5]), np.array([0,0,0]))
saida(np.array([9,5,6]), np.array([1,1,1]))
saida(np.array([9,5,6]), np.array([0,0,0]))
saida(np.array([9,4,7]), np.array([1,1,1]))
saida(np.array([9,4,7]), np.array([0,0,0]))