# FUnção que calcula a probabilidade de theta
def g(theta, params):
    # Se algum dos termos de theta for negativo retorna o valor de -1 assim esse candidato será rejeitado
    if theta[0] < 0 or theta[1] < 0 or theta[2] < 0:
        return -1

    # Se não faz o cálculo da probabilidade utilizando a fórmula
    beta = np.prod(gamma(params))/ gamma(np.sum(params))
    
    resul = np.prod(theta**(params - 1))
    
    return resul

#Função geradora de variáveis aleatórias para uma distribuição de Dirichlet
def thetamcmc(n,x,y):
    
    xy = x+y # Define os parâmetros a_i, i=1,2,3 da função de Dirchlet
    s_xy = sum(xy) # Define o parâmetro a_0, que é a soma dos parâmetros a_i, i=1,2,3
    # Calcula as variâncias de theta_i, i=1,2,3 e também as covariâncias entre elas
    var0 = xy[0]*(s_xy-xy[0])/(s_xy**2*(s_xy+1))
    var1 = xy[1]*(s_xy-xy[1])/(s_xy**2*(s_xy+1))
    var2 = xy[2]*(s_xy-xy[2])/(s_xy**2*(s_xy+1))
    cov01 = -xy[0]*xy[1]/(s_xy**2*2*(s_xy+1))
    cov02 = -xy[0]*xy[2]/(s_xy**2*2*(s_xy+1))
    cov12 = -xy[1]*xy[2]/(s_xy**2*2*(s_xy+1))

    cov = np.array([[var0,cov01,cov02],[cov01,var1,cov12],[cov02,cov12,var2]]) # Define a matriz de covariâncias
    theta_old = np.array([0.3,0.25,0.45]) # Define o chute inicial do algoritmo
    burn_in = 200 # Define o número de iterações para o burn in
    sample = np.zeros((n+burn_in,3)) # Cria a matriz da amostra com os valores de theta
    # Inicia o laço do algoritmo
    for i in range(burn_in+n):
        mean = theta_old # Define a média como o valor anterior de theta
        sample[i] = theta_old # Guarda o valor do theta anterior
        # Define o candidato a ser avaliado
        x_star, y_star, z = np.random.multivariate_normal(mean, cov)
        z_star = 1 - (x_star + y_star)
        theta_star = np.array([x_star, y_star, z_star])

        alpha = np.min([g(theta_star,xy)/g(theta_old,xy),1]) # Define o valor de alpha, probabilidade de aceitação do candidato
        u = np.random.uniform() # Define o parâmetro que diz se o candidato será aceito ou rejeitado
        if u <= alpha:
            theta_old = theta_star # Caso seja aceito o candidato substitui o valor anterior

    sample = sample[burn_in:] # Tira os candidatos calculados no burn in
    
    return sample