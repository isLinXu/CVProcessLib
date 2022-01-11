import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy.random as rd
L = [10,100,200]
p = 0.5
j_0 = 10
show = 1
n_steps = 20000

X = [np.zeros(L[0]),np.zeros(L[1]),np.zeros(L[2])]
P1,P2,P3 = 0,0,0
def Update(X,L,i):
    if i % j_0 == 0:
        X[0] += 1
    piles = np.copy(X)
    for a in range(len(X)):
        if a == 0:
            for b in range(int(piles[a])):
                if rd.random() <= p:
                    X[a+1] += 1
                    X[a] -= 1
        elif a == L-1:
            if piles[a] != 0:
                for b in range(int(piles[a])):
                    if rd.random() <= p:
                        X[a-1] += 1
                        X[a] -= 1
        else:
            if piles[a] != 0:
                for b in range(int(piles[a])):
                    rdm = rd.random()
                    if rdm <= p:
                        X[a+1] += 1
                        X[a] -= 1
                    if rdm > p and rdm <= 2*p:
                        X[a-1] += 1
                        X[a] -= 1

    return X

for i in range(n_steps):
    if i % (n_steps/10) == 0:
        print("%.1f percent done" % (i*100 / n_steps))
    X[0],X[1],X[2] =Update(X[0],len(X[0]),i),Update(X[1],len(X[1]),i),Update(X[2],len(X[2]),i)
fig = plt.figure()
proteins = int(n_steps/j_0)
ax1 = fig.add_subplot(311)
ax1.plot(X[0],color="black")
ax1.set_title("L= %d, proteins = %d" % (len(X[0]),proteins))
ax2 = fig.add_subplot(312)
ax2.plot(X[1],color="black")
ax2.set_title("L= %d, proteins = %d" % (len(X[1]),proteins))
ax3 = fig.add_subplot(313)
ax3.plot(X[2],color="black")
ax3.set_title("L= %d, proteins = %d" % (len(X[2]),proteins))
plt.tight_layout()

plt.show()