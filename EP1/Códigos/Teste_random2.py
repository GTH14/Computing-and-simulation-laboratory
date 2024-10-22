# from random import random
# from math import sqrt, pi
# import matplotlib.pyplot as plt

# def repmat(list_1, n):
#     list_2 = []
#     l= len(list_1)
#     for j in range(l):
#         for i in range(n):
#             list_2.append(list_1[j])
#     return list_2
# list_n = [ 40000, 50000, 60000, 70000, 80000, 90000, 100000, 200000, 250000, 500000]
# lista_teste = 3*list_n
# list_A =[]
# list_A1 = []
# list_somaT = []
# list_mediaT = []
# list_err_A = []
# list_err_A1 = []
# list_err_mediaT = []
# l_n = len(list_n)
# n_r = 5
# #n_h = 5

# for j in range(l_n):
#     n = list_n[j]
#     sT= 0
#     for i in range(n_r):
#         Axn = 0
#         for k in range(n) :
#             x = random()
#             y = random()
#             if sqrt((x-0.5)**2 + (y-0.5)**2) <= 0.5:
#                 T = 1
#             else:
#                 T = 0
#             Axn = Axn + T
#         A = Axn/n
#         err_A = 100*(A-pi/4)/(pi/4)
#         list_A.append(A)
#         list_err_A.append(err_A)
        
#         sT = sT + A
#     list_somaT.append(sT)
#     media_sT = sT/n_r
#     list_mediaT.append(media_sT)
#     err_mediaT = 100*(media_sT - pi/4)/(pi/4)
#     list_err_mediaT.append(err_mediaT)

# ##print("Valor estimado da área: ", A)
# ##print("Valor correto: ", pi/4)
# ##err = A-pi/4
# ##print("err: ", err) """

# for j in range(l_n):
#     n1 = list_n[j]
#     Axn1 = 0
#     for i in range(n_r*n1):
        
#         x = random()
#         y = random()
#         if sqrt((x-0.5)**2 + (y-0.5)**2) <= 0.5:
#             T1 = 1
#         else:
#             T1 = 0
#         Axn1 = Axn1 + T1
#     A1 = Axn1/(n1*n_r)
#     err_A1 = 100*(A1-pi/4)/(pi/4)
#     list_A1.append(A1)
#     list_err_A1.append(err_A1)
# list_plot = repmat(list_n, n_r)
# #list_plot1 = repmat(list_n, n_h)
# ##print(list_plot)
# ##print(list_A ,"\n")
# ##print(list_err_A, "\n")
# ##print(list_somaT, "\n")
# ##print(list_mediaT, "\n")
# ##print(list_err_mediaT, "\n")
# print(lista_teste)
# plt.figure(1)
# plt.plot(list_n, list_err_mediaT, 'ro')
# plt.plot(list_n, list_err_A1, 'bo')

# plt.figure(2)
# plt.plot(list_plot, list_err_A, 'ro')


# plt.show()


# x = 2.55
# y = 3
# z = y/(0.0005*x)
# print(z)
# from IPython.display import display

# x = [1, 2, 3, 4]
# display("Variável x x)

y = [1, 2, 3]
x = y-[1,1,1]
print(x)