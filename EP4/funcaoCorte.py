def cortes(k,theta,x,y,ff):
    n = len(theta)
    v0 = 0
    vk = np.max(ff)
    v = []
    v.append(v0)
    dv = (vk-v0)/k
    i = 1
    while i<k:
        ui_1 = u(v[i-1],n,theta,x,y)
        ui = u(v[i-1]+dv,n,theta,x,y)
        if (ui-ui_1)-1/k>0:
            j = 1
            ub = ui
            eps = dv/100
            while (ub-ui_1)-1/k>0:
                vb = v[i-1]+dv-eps*j
                ub = u(vb,n,theta,x,y)
                j += 1
            vi = vb
        elif (ui-ui_1)-1/k<0:
            j = 1
            ub = ui
            eps = dv/100
            while (ub-ui_1)-1/k<0:
                vb = v[i-1]+dv+eps*j
                ub = u(vb,n,theta,x,y)
                j += 1
            vi = vb
        else:
            vi = v[i-1]+dv
        v.append(vi)
        i += 1
    v.append(vk)
    return v