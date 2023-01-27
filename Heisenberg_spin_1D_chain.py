import numpy as np
import scipy
import matplotlib.pyplot as plt 
import matplotlib as mpl
from functools import reduce
from itertools import repeat, chain

def plot_bands(energies, title, figsize, s, ticks):
    x = list(range(len(energies)))
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, energies, c = 'black', s=s, marker="_", linewidth=2, zorder=3)
    tick_spacing = 1
    if ticks == False: 
        ax.set_xticks([])
    else:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(tick_spacing))
    ax.grid(axis='y')
    ax.margins(0.1)
    ax.set_xlabel('index')
    ax.set_ylabel('E')
    ax.set_title(title)


def S_site(index, N, S):
    size = len(S)
    chain_I = chain([np.identity(size**(index))], [S], [np.identity(size**(N - index))])
    return reduce(np.kron, chain_I)

if __name__ == "__main__": 

    S_plus = np.sqrt(2) * np.array([[0,1,0],
                                    [0,0,1],
                                    [0,0,0]])

    S_minus = np.sqrt(2) * np.array([[0,0,0],
                                    [1,0,0],
                                    [0,1,0]])
    S_z = np.array([[1,0,0],
                    [0,0,0],
                    [0,0,-1]])

    I = np.array([[1,0,0],
                    [0,1,0],
                    [0,0,1]])

    N = 15  #10 is to much for M1 macos ;) 
    H = 0
    for index in range(N):
        H += 1/2 * (np.dot(S_site(index, N, S_plus),S_site(index+1, N, S_minus)) \
            + np.dot(S_site(index, N, S_minus),S_site(index+1, N, S_plus))) + np.dot(S_site(index, N, S_z),S_site(index+1, N, S_z))

    energies, vectors = np.linalg.eigh(H)
    
    np.savetxt("spin_energies.csv", energies, delimiter=" ")

    #plot_bands(energies, title = "s=1, " + str(N+1) +" sites, open", figsize=(10,10),s=50, ticks = False)
    #plt.axis([-2, 18, -5, -2])
    #print(energies[0:15])
    #plt.show()
    #plt.savefig("plot.png")