## Descobre os melhores parâmetros para a distribuução Beta

# Importa as funções utilizadas na rotina
from math import exp, cos
from scipy.stats import beta
from time import time
# Define a funções
a = 0.54114733
b = 0.11260680
def f(x): # Função que se quer integrar
    return exp(-a*x)*cos(b*x)

def g(x, alfa, betha): # Função distribuição de probalidade Beta
    return beta.pdf(x,alfa, betha, loc=0, scale=1)

def Calc_Err_Quad(tup): # Função que calcula o erro quadrático entre os pontos de beta e da função f(x)
    N = 2*10**3 # Número de pontos utilizados
    i = 10 # Tira os 10 primeiros pontos para não dar infinito
    alfa = tup[0] # Define o parâmetro alfa da função Beta através da tupla parâmetro desta função
    betha = tup[1] # Define o parâmetro beta da função Beta através da tupla parâmetro desta função
    Err_quad = 0 # Inicializa a variável do erro quadrático

    while i<N:
        Err_quad = Err_quad + (f(i/N)-g(i/N, alfa, betha))**2 # Calcula o erro quadrático
        i +=1
    Err_quad = Err_quad/N
    return Err_quad

def Procura_alpha_beta(list_tup): # Função procura quais são os melhores parâmetros alfa e beta
    l = len(list_tup) # Descobre quantas tuplas foram definidas
    i = 1 # Contador do loop
    trocou = 0
    escolha = list_tup[0] # Variável que conterá o melhor par alfa e beta
    Err_quad_escolha = Calc_Err_Quad(escolha) # Define o erro quadrático da primeira tupla
    while i < l-1:
        Err_quad_tup_i = Calc_Err_Quad(list_tup[i])
        if Err_quad_escolha>Err_quad_tup_i: # Condiciona que para trocar a tupla vigente
        # o erro quadrático dessa nova tupla deve ser menor que o da tupla
            escolha = list_tup[i] # Troca para a tupla com menor erro quadrático    
            Err_quad_escolha = Err_quad_tup_i # Troca o erro quadrático para o da nova tupla
        i +=1
    return escolha

def Sort_tup(n , d, v_inicial): # Função que cria uma lista de tuplas para o alfa e o beta
# os parâmetros de entra dessa função são n o número de tuplas, d o incremento que os parâmetros 
# recebem e v_inicial é a tupla que contém o valor inicial que os parâmetros alfa e beta terão
# inicialmente 
    i = 0 # Contador do primeiro loop
    list_tup = [] # Inicializa a lista das tuplas
    while i<n:
        j = 0
        alfa = v_inicial[0]+i*d # Adiciona o incremento para o parâmetro alfa
        while j<n:
            betha = v_inicial[1]+j*d # Adiciona o incremento para o parâmetro beta
            list_tup.append((alfa,betha))
            j+=1
        i+=1
    return list_tup

list_tup = Sort_tup(12, 0.05, (0.5,0.5))

escolha = Procura_alpha_beta(list_tup)

print(escolha)
