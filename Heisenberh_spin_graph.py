
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
from functools import reduce
from itertools import chain, product
from scipy.linalg import logm


class Graph(object):

    # Adjacency Matrix representation in Python

    # Initialize the matrix
    def __init__(self, size):
        self.adjMatrix = np.zeros((size,size))
        self.size = size

    # Add edges
    def add_edge(self, v1, v2):
        if v1 == v2:
            print("Same vertex %d and %d" % (v1, v2))
        self.adjMatrix[v1][v2] = 1
        self.adjMatrix[v2][v1] = 1

    # Remove edges
    def remove_edge(self, v1, v2):
        if self.adjMatrix[v1][v2] == 0:
            print("No edge between %d and %d" % (v1, v2))
            return
        self.adjMatrix[v1][v2] = 0
        self.adjMatrix[v2][v1] = 0

    def __len__(self):
        return self.size
    
    def adjMatrix(self): 
        return self.adjMatrix

class Heisenberg(object):

    #Creating Heisenberg Hamiltonian 

    #Initialization of the system
    def __init__(self,N) -> None:
        self.size_of_system = N
        self.chain_I = []
        self.energies = []
        self.vectors = []
        self.possible_basis = []
        self.H = 0

    #Using tensor product to calculate S_i matrix
    def S_site(self, index, S):
        N = self.size_of_system
        self.chain_I = chain([np.identity(len(S)**(index))], [S], [np.identity(len(S)**(N - index))])
        return reduce(np.kron, self.chain_I)

    #definition of S matrices and diagonalization
    def diagonalize_Hamiltonian(self, adjMatrix):
        
        """
        matrices for S = 1
        

        S_plus = np.sqrt(2) * np.array([[0,1,0],
                                    [0,0,1],
                                    [0,0,0]])

        S_minus = np.sqrt(2) * np.array([[0,0,0],
                                    [1,0,0],
                                    [0,1,0]])
        S_z = np.array([[1,0,0],
                    [0,0,0],
                    [0,0,-1]])
        """
        '''
        matrices for S = 1/2
        '''
        
        S_plus = np.array([[0,1],
                            [0,0]])

        S_minus = np.array([[0,0],
                            [1,0]])
        
        S_z = 1/2* np.array([[1,0],
                            [0,-1]])
        

        #using adjacency matrix to define neighbouring sites
        for i in range(len(adjMatrix)):
            for j in range(len(adjMatrix)):
                if adjMatrix[j][i] == 1:
                    self.H += 1/2 * (np.dot(self.S_site(j, S_plus),self.S_site(i, S_minus)) \
                    + np.dot(self.S_site(j, S_minus),self.S_site(i, S_plus))) \
                    + np.dot(self.S_site(j, S_z), self.S_site(i, S_z))
        self.energies, self.vectors = np.linalg.eigh(self.H)
        print(len(self.H))
        return self.energies, self.vectors
    
    def s_z_operator(self, S_z, eigenvector):
        return np.dot(S_z,eigenvector)
    
    def calculate_basis(self):
        N = self.size_of_system
        for i in range(N+1):
            #for bais s=1/2 -> (up - True, down - False)
            self.possible_basis.append([True,False])
            #for bais s=1 -> (-1,0,1)
            #self.possible_basis.append([-1,0,1])
        return list(product(*self.possible_basis))
    
    def calculate_rho(self,n):
        #print("Eigenenergy is: ", self.energies[n])
        #print("Eigenvector is: ", self.vectors[n])
        #print(len(self.vectors[n]))
        return np.kron(self.vectors[n],self.vectors[n].conj()).reshape(len(self.vectors[n]),len(self.vectors[n]))
    
    def calculate_reduced_rho_2_spin(self,rho_big):
        #system 3 spinów, liczymy dla rho_2 (traceout spin 1 i 3)
        rho = np.zeros((2,2),dtype = complex)
        rho[0,0] = rho_big[0,0]+rho_big[1,1]+rho_big[4,4]+rho_big[5,5]
        rho[0,1] = rho_big[0,2]+rho_big[1,3]+rho_big[4,6]+rho_big[5,7]
        rho[1,0] = rho_big[2,0]+rho_big[3,1]+rho_big[6,4]+rho_big[7,5]
        rho[1,1] = rho_big[2,2]+rho_big[3,3]+rho_big[6,6]+rho_big[7,7]
        return rho
    
    def calculate_reduced_rho_2_spin_env(self,rho_big):
        #system 3 spinów, liczymy dla rho_2 (traceout spin 2)
        rho = np.zeros((4,4),dtype = complex)
        rho[0,0] = rho_big[0,0] + rho_big[2,2]
        rho[0,1] = rho_big[0,1] + rho_big[2,3]
        rho[0,2] = rho_big[0,4] + rho_big[2,6]
        rho[0,3] = rho_big[0,5] + rho_big[2,7]
        
        rho[1,0] = rho_big[1,0] + rho_big[3,2]
        rho[1,1] = rho_big[1,1] + rho_big[3,3]
        rho[1,2] = rho_big[1,4] + rho_big[3,6]
        rho[1,3] = rho_big[1,5] + rho_big[3,7]
        
        rho[2,0] = rho_big[4,0] + rho_big[6,2]
        rho[2,1] = rho_big[4,1] + rho_big[6,3]
        rho[2,2] = rho_big[4,4] + rho_big[6,6]
        rho[2,3] = rho_big[4,5] + rho_big[6,7]
        
        rho[3,0] = rho_big[5,0] + rho_big[7,2]
        rho[3,1] = rho_big[5,1] + rho_big[7,3]
        rho[3,2] = rho_big[5,4] + rho_big[7,6]
        rho[3,3] = rho_big[5,5] + rho_big[7,7]
        
        return rho
    
    def calculate_entropy(self,rho_reduced,n):
        #entropy = -np.trace(rho_reduced * logm(rho_reduced))
        #n - number of spins 
        eigen_rho, vectors = np.linalg.eigh(rho_reduced)
        #entropy = -sum(np.log(eigen_rho, out=np.zeros_like(eigen_rho), where=(eigen_rho!=0.0)))
        entropy = -sum(np.log(eigen_rho, where=0<eigen_rho, out=0.0*eigen_rho))

        '''
        for i in range(len(eigen_rho)): 
            print(eigen_rho[i])
            if eigen_rho[i] <= 0.0:
                entropy += 0.0
            else:
                entropy += -(np.log(eigen_rho[i]))
        '''
        return entropy/n

    #Plotting energy bands with index
    #title -> header of plot, 
    #figsize -> size of figure, 
    #s -> thickness of a band 
    #ticks -> True for showing ticks on x axis 
    def plot_bands(self, title, figsize, s, ticks):
        x = list(range(len(self.energies)))
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(x, self.energies, c = 'black', s=s, marker="_", linewidth=2, zorder=3)
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
        
    def plot_entropy(self, entropy, color, title, figsize, s):
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(self.energies, entropy, c = color , s=s) #, marker="_", linewidth=2, zorder=3)
        ax.grid(axis='y')
        ax.margins(0.1)
        ax.set_xlabel('Energy')
        ax.set_ylabel('Entropy')
        ax.set_title(title)
 
def main(N,adjMatrix): #N+1 -> size of graph

    H = Heisenberg(N)
    H.diagonalize_Hamiltonian(adjMatrix)
    #H.plot_bands(title = "s=1, " + str(N+1) +" sites, graph", figsize=(10,12),s=100, ticks = False)
    H.plot_bands(title = "s=1/2, " + str(N+1) +" sites, graph", figsize=(10,12),s=100, ticks = True)
    basis = H.calculate_basis()
    print("Our basis is: ", basis)

    entropy_all_system = []
    entropy_all_env = []
    for n in range(8):
        rho_sys = H.calculate_rho(n)
        rho_2 = H.calculate_reduced_rho_2_spin(rho_sys)
        rho_env = H.calculate_reduced_rho_2_spin_env(rho_sys)
        #print(np.trace(rho_2))
        #print(np.trace(rho_env))
        if not .9999999999 <= np.trace(rho_2) <= 1.000000001:
            print(np.trace(rho_2))
            print(np.trace(rho_env))
            
        entropy_sys = H.calculate_entropy(rho_2,1)
        entropy_all_system.append(entropy_sys)
        entropy_env = H.calculate_entropy(rho_env,2)
        entropy_all_env.append(entropy_env)
    
    H.plot_entropy(entropy_all_system, color = 'red', title = "s=1/2, " + str(N+1) +" sites, system ", figsize=(10,12),s=50)
    H.plot_entropy(entropy_all_env, color = 'blue', title = "s=1/2, " + str(N+1) +" sites, environment ", figsize=(10,12),s=50)
    
    #plt.axis([-1, 18, -10, -7.0])
    plt.show()

if __name__ == '__main__':

    ######## Heisenberg Graph #########
    '''                                                                    
    Program for calculating and diagonalization of Heisenberg Hamiltonian for graph with defined adjacency matrix                               # 
    # class Graph - define graph, class Heisenberg - calculation of H
    # here you can creating a graph using add_edge methods or declare prepared matrix (adjMatrix).                                              #
    # N is a size of the system # 
    '''
    
    '''g = Graph(N+1)
    g.add_edge(0, 1)
    g.add_edge(0, 4)
    g.add_edge(1, 4)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(4, 3)
    g.add_edge(3, 5)
    adjMatrix = g.adjMatrix
    print(adjMatrix)
    '''
    #adjMatrix = np.array([[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1],[1,0,0,0,0,0,0]])
    adjMatrix = np.array([[0,1,0],[0,0,1],[1,0,0]])
    N = len(adjMatrix) - 1
    N = 3
    
    main(N, adjMatrix)
    print("Success")

