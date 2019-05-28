import numpy as np

def f(x, c):
    return 0.5*c*x + 2.0*(1-c)*x + 1.0

N = 100

x = np.random.normal(0, 1.0, N)
c = np.random.randint(2, size=N)
y = f(x, c) + np.random.normal(0, 0.2, N)

with open('out.csv', 'w') as f:
    f.write('y,x,c\n')
    for i in range(N):
        f.write('%.2f,%.2f,%s\n' % (y[i], x[i], ['A','B'][c[i]]))
