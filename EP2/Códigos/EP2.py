# Importa as funções utilizadas nesse código
from random import  seed, random, betavariate
from math import cos, exp,sqrt
from time import time
from scipy.stats import t, beta

# Define os parâmetros de entrada
a = 0.541147330 # 0.RG
b = 0.42490023810 #0.CPF
alfa = 0.9 # Parâmetro alfa para a função Beta
betha = 1.05 # Parâmetro beta para a função Beta
Integral_exp = 1/a*(1-exp(-a)) # Valor da integral de 0 a 1 de exp(-ax)
Integral_1 = 1-a/2  # Valor da integral de 0 a 1 de 1-ax, Polinômio de Taylor de grau 1

# Define as funções
def Print_info(media, n, err, t): # Função para printar os resultados obtidos no método
    print("Media: ", media)
    print("O numero de iterações: ", n)
    print("O erro: ", err*100, "%")
    print("Tempo de simulação: ", t,"\n")

def f(x): # Função que se quer integrar
    return exp(-a*x)*cos(b*x)

def T(x,y): # Função que testa se y<=f(x)
    if y <= f(x):
        return 1
    else:
        return 0

def g(x): # Função distribuição de probabilidade da variável aleatória
    return beta.pdf(x,alfa, betha, loc=0, scale=1)

def phi1(x): # Função utilizada como variável de controle 
    return exp(-a*x)

def phi2(x): # Função utilizada como variável de controle 
    return 1-a*x

###### Crude Monte Carlo Method
seed(1) # Define uma seed para os resultados obtidos pelo aluno 
#sejam os mesmos obtidos pelo monitor

# Define o método Crude para estimar o valor da integral
def Crude_MC(): 
    n = 0 # Inicializa a variável n que representa o número de iterações necessária
    #para obter um erro menor que 0.05%
    soma = 0 # Inicializa o somatório de f(x_i) de i = 1 até n
    soma_quad = 0 # Inicializa o somatório de f(x_i)^2 de i = 1 até n
    err = 1 # Inicializa a variável do erro relativo
    t_alpha = 2.576 # Define o valor do t_Student
    t1 = time() # Utilizado para calcular o tempo para rodar a função
    while err > 0.0005: # Só sairá do loop se o erro relativo for menor que 0.0005 ou 0.05%
        n +=1
        x1 = random() # Atribui o valor a variável aleatória x_i
        f1 = f(x1) # Calcula f(x_i)
        soma = soma + f1 # Adiciona termo ao somatório de f(x_i)
        soma_quad = soma_quad + f1**2 # Adiciona termo ao somatório de f(x_i)^2

        if n == 1: # Condição para que não dê erro na primeira roda pois há um divisão por n-1
        # e quando n = 1 haverá divisão por zero, ou seja, dará um erro
            err = 1
        else:
            var = (soma_quad - soma**2/n)/(n-1) # Calcula a variância do método
            err = t_alpha*sqrt(var/n) # Calcula o erro desse método para dado n
    t2 = time()
    media = soma/n # Calcula o estimador para a integral de f(x)
    return media, n, err, t2-t1 # Retorna a estimativa, o n, o erro e o tempo para rodar a rotina

media_c, n_c, err_c, t_c = Crude_MC()
print("Crude Monte Carlo Method:")
Print_info(media_c, n_c, err_c, t_c)


###### Hit or Miss Monte Carlo Method
seed(10) # Define uma seed para os resultados obtidos pelo aluno 
#sejam os mesmos obtidos pelo monitor

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
media_h, n_h, err_h, t_h = Hit_Miss_MC()
print("Hit or Miss Carlo Method:")
Print_info(media_h, n_h, err_h, t_h)


###### Importance Sampling Monte Carlo Method
seed(21) # Define uma seed para os resultados obtidos pelo aluno 
#sejam os mesmos obtidos pelo monitor

# Define o método Importance Sampling para estimar o valor da integral
def Importance_Sampling():
    n = 0 # Inicializa a variável n que representa o número de iterações necessária
    #para obter um erro menor que 0.05%
    soma = 0 # Inicializa o somatório de f(x_i)/g(x_i) de i = 1 até n
    soma_quad = 0 # Inicializa o somatório de (f(x_i)/g(x_i))^2 de i = 1 até n
    t_alpha = 2.576 # Define o valor do t_Student
    err = 1 # Inicializa a variável do erro relativo
    t1 = time() # Utilizado para calcular o tempo para rodar a função
    while err > 0.0005: # Só sairá do loop se o erro relativo for menor que 0.0005 ou 0.05%
        n += 1
        x1 = betavariate(alfa,betha) # Atribui o valor a variável aleatória x_i
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
            err = -t.ppf(0.005,n-1)*sqrt(var/n) # Calcula o erro desse método para dado n
    t2 = time()
    media = soma/n # Calcula o estimador para a integral de f(x)
    return media, n, err, t2-t1 # Retorna a estimativa, o n, o erro e o tempo para rodar a rotina

media_s, n_s, err_s, t_s = Importance_Sampling()
print("Importance Sampling Monte Carlo Method:")
Print_info(media_s, n_s, err_s, t_s)


###### Control Variate Monte Carlo Method
seed(81) # Define uma seed para os resultados obtidos pelo aluno 
#sejam os mesmos obtidos pelo monitor

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
        x = random() # Atribui o valor a variável aleatória x_i
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
            err = -t.ppf(0.005,n-1)*sqrt(var/n) # Calcula o erro desse método para dado n
    t2 = time()
    media = (soma_f-soma_phi)/n + Integral_known # Calcula o estimador para o valor da integral de f(x)
    return media, n, err, t2-t1 # Retorna a estimativa, o n, o erro e o tempo para rodar a rotina

media1, n1, err1, t1 = Control_Variate(f,phi1, Integral_exp) # Utiliza phi(x) = exp(-a*x)
media2, n2, err2, t2 = Control_Variate(f,phi2, Integral_1) # Utiliza phi(x) = 1 - a*x
print("Control Variate Monte Carlo Method:")
print("phi(x) = exp(-a*x) :")
Print_info(media1, n1, err1, t1)
print("phi(x) = 1-a*x :")
Print_info(media2, n2, err2, t2)