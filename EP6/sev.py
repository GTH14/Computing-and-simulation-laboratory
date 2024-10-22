# import scipy.stats as st
import numpy as np
# def sev(ev):
#     ev_barra = 1-ev
#     aux = st.chi2.ppf(ev_barra, 3, loc=0, scale=1)
#     sev_barra = st.chi2.cdf(aux, 1, loc=0, scale=1)
#     sev1 = 1-sev_barra
#     return sev1
list_ijk = np.array([[1,2,3],[2,2,3],[5,1,2]])
# for i, j, k in zip(list_ijk[0],list_ijk[1], list_ijk[2]):
for i,j,k in list_ijk:
    print(i)
    print(j)
    print(k)
np.savetxt("K1.csv", list_ijk, delimiter=" , ")