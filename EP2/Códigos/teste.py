import chaospy
for i in range(6):
    x = chaospy.create_halton_samples(1000000,1, -1+i)
    print(x)