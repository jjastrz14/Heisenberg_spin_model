# version 2.0 for spin 1D chain diagonalization (max number of sites 8-10) using numpy and:
# -> calculating a density matrix
# -> calculating a reduced density matrix of chain divided into two equal subsytems
# -> calculating entropy of this subsystems
####

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
from functools import reduce
from itertools import chain, product, repeat
from scipy.linalg import logm
import os 


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
    def __init__(self,N, S, directory = None) -> None:
        self.size_of_system = N
        self.chain_I = []
        self.energies = []
        self.vectors = []
        self.possible_basis = []
        self.H = 0
        
        if S == 1:
        #matrices for S = 1
            self.S_plus = np.sqrt(2) * np.array([[0,1,0],
                                            [0,0,1],
                                            [0,0,0]])

            self.S_minus = np.sqrt(2) * np.array([[0,0,0],
                                            [1,0,0],
                                            [0,1,0]])
            self. S_z = np.array([[1,0,0],
                            [0,0,0],
                            [0,0,-1]])
        
        
        elif S == 1/2:
            #matrices for S = 1/2
            self.S_plus = np.array([[0,1],
                                    [0,0]])

            self.S_minus = np.array([[0,0],
                                    [1,0]])
        
            self.S_z = 1/2* np.array([[1,0],
                                    [0,-1]])
            self.I = np.array([[1,0],
                                [0,1]])
        

    #Using tensor product to calculate S_i matrix
    def S_site(self, index, S):
        N = self.size_of_system
        self.chain_I = chain([np.identity(len(S)**(index))], [S], [np.identity(len(S)**(N - index))])
        return reduce(np.kron, self.chain_I)
    
    def S_z_operator(self):
        #calculating S_z operator as sum S_z_1 + S_z_2 + ... + S_z_N
        S_z_operator = 0
        for i in range(self.size_of_system+1):
            S_z_operator  += self.S_site(i, self.S_z)
            
        return S_z_operator
        
    def calc_Sz(self, eigenvector):
        # Calculate the conjugate transpose of the eigenvector
        psi_dagger = np.conj(eigenvector.T)
        # Calculate the expectation value of S_z
        Sz_total = np.dot(psi_dagger, np.dot(self.S_z_operator(), eigenvector))
        return Sz_total
    
    def eig_diagonalize(self,A):
        #fucntion for diagonalization with sorting eigenvalues and rewriting eigenvectors as a list
        eigenValues, eigenVectors = np.linalg.eig(A)
        idx = eigenValues.argsort()
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        return eigenValues, eigenVectors
    
    def diagonalize_Hamiltonian(self, adjMatrix):
        #definition of S matrices and diagonalization

        #using adjacency matrix to define neighbouring sites
        for i in range(len(adjMatrix)):
            for j in range(len(adjMatrix)):
                if adjMatrix[j][i] == 1:
                    self.H += 1/2 * (np.dot(self.S_site(j, self.S_plus),self.S_site(i, self.S_minus)) \
                    + np.dot(self.S_site(j, self.S_minus),self.S_site(i, self.S_plus))) \
                    + np.dot(self.S_site(j, self.S_z), self.S_site(i, self.S_z))
                    
        self.energies, self.vectors = self.eig_diagonalize(self.H)
        print("Len of Hamiltonian: ", len(self.H))
        return self.energies, self.vectors
    
    def normalization_of_energies(self,vector_of_numbers):
        #normalization of list containing negative values
        minimum = np.amin(vector_of_numbers)
        vector_of_numbers = vector_of_numbers + abs(minimum)
        return vector_of_numbers/sum(vector_of_numbers)

    def calculate_basis(self):
        #probably not intresting 
        N = self.size_of_system
        for i in range(N+1):
            #for bais s=1/2 -> (up - True, down - False)
            self.possible_basis.append([True,False])
            #for bais s=1 -> (-1,0,1)
            #self.possible_basis.append([-1,0,1])
        return list(product(*self.possible_basis))

    def calculate_rho(self,n):
        # n -> is the interator over eigenvectors
        return np.kron(self.vectors[:,n],self.vectors[:,n].conj()).reshape(len(self.vectors[:,n]),len(self.vectors[:,n]))
    
    
    def calculate_reduced_rho_sys(self,rho_big, spin, sites_in_subsystem): 
        
        number_of_states = int((2*spin + 1)**sites_in_subsystem)
        rho_sys = np.zeros((number_of_states,number_of_states),dtype = complex)
        #system 
        for i in range(number_of_states):
            j = i * number_of_states
            for k in range(number_of_states):
                rho_sys[k, i] = sum(rho_big[k * number_of_states + l, j + l] for l in range(number_of_states))
            
        return rho_sys
    
    def calculate_reduced_rho_env(self,rho_big, spin, sites_in_subsystem):
        
        number_of_states = int((2*spin + 1)**sites_in_subsystem)
        rho_env = np.zeros((number_of_states,number_of_states),dtype = complex)
        
        #env   
        for i in range(number_of_states):
            for j in range(number_of_states):
                rho_env[j,i] = sum(rho_big[j + l * number_of_states ,i + l * number_of_states] for l in range(number_of_states))
        
        return rho_env
    
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
    
    def calculate_reduced_rho_4_spin_sys(self,rho_big):
        
        #system 4 spinów, liczymy dla (traceout spin 3,4)
        rho = np.zeros((4,4),dtype = complex)
        
        rho[0,0] = rho_big[0,0] + rho_big[1,1] + rho_big[2,2] + rho_big[3,3]
        rho[0,1] = rho_big[0,4] + rho_big[1,5] + rho_big[2,6] + rho_big[3,7]
        rho[0,2] = rho_big[0,8] + rho_big[1,9] + rho_big[2,10] + rho_big[3,11]
        rho[0,3] = rho_big[0,12] + rho_big[1,13] + rho_big[2,14] + rho_big[3,15]
        
        rho[1,0] = rho_big[4,0] + rho_big[5,1] + rho_big[6,2] + rho_big[7,3]
        rho[1,1] = rho_big[4,4] + rho_big[5,5] + rho_big[6,6] + rho_big[7,7]
        rho[1,2] = rho_big[4,8] + rho_big[5,9] + rho_big[6,10] + rho_big[7,11]
        rho[1,3] = rho_big[4,12] + rho_big[5,13] + rho_big[6,14] + rho_big[7,15]
        
        rho[2,0] = rho_big[8,0] + rho_big[9,1] + rho_big[10,2] + rho_big[11,3]
        rho[2,1] = rho_big[8,4] + rho_big[9,5] + rho_big[10,6] + rho_big[11,7]
        rho[2,2] = rho_big[8,8] + rho_big[9,9] + rho_big[10,10] + rho_big[11,11]
        rho[2,3] = rho_big[8,12] + rho_big[9,13] + rho_big[10,14] + rho_big[11,15]
        
        rho[3,0] = rho_big[12,0] + rho_big[13,1] + rho_big[14,2] + rho_big[15,3]
        rho[3,1] = rho_big[12,4] + rho_big[13,5] + rho_big[14,6] + rho_big[15,7]
        rho[3,2] = rho_big[12,8] + rho_big[13,9] + rho_big[14,10] + rho_big[15,11]
        rho[3,3] = rho_big[12,12] + rho_big[13,13] + rho_big[14,14] + rho_big[15,15]
        
        return rho
    
    def calculate_reduced_rho_4_spin_env(self,rho_big):
        
        #system 4 spinów, liczymy dla (traceout spin 1,2)
        rho = np.zeros((4,4),dtype = complex)
        
        rho[0,0] = rho_big[0,0] + rho_big[4,4] + rho_big[8,8] + rho_big[12,12]
        rho[0,1] = rho_big[0,1] + rho_big[4,5] + rho_big[8,9] + rho_big[12,13]
        rho[0,2] = rho_big[0,2] + rho_big[4,6] + rho_big[8,10] + rho_big[12,14]
        rho[0,3] = rho_big[0,3] + rho_big[4,7] + rho_big[8,11] + rho_big[12,15]
        
        rho[1,0] = rho_big[1,0] + rho_big[5,4] + rho_big[9,8] + rho_big[13,12]
        rho[1,1] = rho_big[1,1] + rho_big[5,5] + rho_big[9,9] + rho_big[13,13]
        rho[1,2] = rho_big[1,2] + rho_big[5,6] + rho_big[9,10] + rho_big[13,14]
        rho[1,3] = rho_big[1,3] + rho_big[5,7] + rho_big[9,11] + rho_big[13,15]
        
        rho[2,0] = rho_big[2,0] + rho_big[6,4] + rho_big[10,8] + rho_big[14,12]
        rho[2,1] = rho_big[2,1] + rho_big[6,5] + rho_big[10,9] + rho_big[14,13]
        rho[2,2] = rho_big[2,2] + rho_big[6,6] + rho_big[10,10] + rho_big[14,14]
        rho[2,3] = rho_big[2,3] + rho_big[6,7] + rho_big[10,11] + rho_big[14,15]
        
        rho[3,0] = rho_big[3,0] + rho_big[7,4] + rho_big[11,8] + rho_big[15,12]
        rho[3,1] = rho_big[3,1] + rho_big[7,5] + rho_big[11,9] + rho_big[15,13]
        rho[3,2] = rho_big[3,2] + rho_big[7,6] + rho_big[11,10] + rho_big[15,14]
        rho[3,3] = rho_big[3,3] + rho_big[7,7] + rho_big[11,11] + rho_big[15,15]
        
        return rho
    
    def calculate_entropy(self,rho_reduced,n):
        #n - number of spins in the subsystem
        eigen_rho, vectors = self.eig_diagonalize(rho_reduced)  
        entropy = -sum(eigen_rho*np.log(eigen_rho, where=0<eigen_rho, out=0.0*eigen_rho))
        '''
        entropy = 0
        for i in range(len(eigen_rho)):
            print(eigen_rho[i])
            if eigen_rho[i] <= 0.0:
                entropy += 0.0
            else:
                entropy += -(eigen_rho[i]*np.log(eigen_rho[i]))
        '''
        return entropy/(n*np.log(n)), eigen_rho
    
    def calculate_S_z(self): 
        #S_z value calculated as inner product of S_z operator and eigenvectors of H
        S_z_total = []
        for i in range(len(self.vectors)):
            S_z_total.append(self.calc_Sz(self.vectors[:,i]))
            
        return S_z_total
    
    
class Plots(object):
        #class for plotting
        
        def __init__(self, S, energies, directory = None) -> None:
            
            self.energies = energies 
            
            if S == 1:
                self.s_number = 's=1'
            elif S == 1/2:
                self.s_number = 's=1_2'
            
            if directory is not None:
                self.directory = os.path.join('./', directory)
                os.makedirs(directory, exist_ok=True)
            else:
                self.directory = './results_'+self.s_number
  
        def plot_bands(self, title, figsize, s, ticks, suffix):
            
            #Plotting energy bands with index
            #title -> header of plot, 
            #figsize -> size of figure, 
            #s -> thickness of a band 
            #ticks -> True for showing ticks on x axis 
            x = list(range(len(self.energies)))
            #energies_normalized = self.normalization_of_energies(self.energies)
            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(x, self.energies, c = 'black', s=s, marker="_", linewidth=5, zorder=3)
            tick_spacing = 1
            if ticks == False: 
                ax.set_xticks([])
            else:
                ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(tick_spacing))
            ax.grid(axis='y')
            ax.margins(0.1)
            ax.set_xlabel('index')
            ax.set_ylabel('E')
            ax.set_title(self.s_number + title)
            
            filename = 'bands_'
            if suffix is not None:
                filename += suffix
            plt.savefig(os.path.join(self.directory, filename + '.png'), bbox_inches='tight', dpi=200)
            plt.close()
            
            
        def plot_bands_with_s_z(self, s_z_values, title, figsize, s, ticks, suffix):
        
            x = list(range(len(self.energies)))
            fig, ax = plt.subplots(figsize=figsize)
            
            ax.scatter(x, self.energies, c = 'black', s=s, marker="_", linewidth=5, zorder=3)
            
            for i, txt in enumerate(s_z_values):
                ax.annotate(txt, (x[i], self.energies[i]), xytext = (x[i] - 0.2, self.energies[i] + 0.05))
                ax.annotate("$S_z$", (x[i], self.energies[i]), xytext = (x[i] - 0.2, self.energies[i] - 0.07))
                
            tick_spacing = 1
            if ticks == False: 
                ax.set_xticks([])
            else:
                ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(tick_spacing))
            ax.grid(axis='y')
            ax.margins(0.1)
            ax.set_xlabel('index')
            ax.set_ylabel('E')
            ax.set_title(self.s_number + title)
            
            filename = 'bands_'
            if suffix is not None:
                filename += suffix
            plt.savefig(os.path.join(self.directory, filename + '.png'), bbox_inches='tight', dpi=200)
            plt.close()
            
        def plot_entropy(self, entropy, color, title, figsize, s, suffix):
            # plotting - entropia dla odpowiedniej eigenenergii H 
            #entropy = self.normalization_of_entropy(entropy)
            
            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(self.energies, entropy, c = color , s=s , marker="_", linewidth=5, zorder=3)
            ax.grid(axis='y')
            ax.margins(0.1)
            start, end = ax.get_ylim()
            ax.yaxis.set_ticks(np.arange(start, end, 0.05))
            #ax.set_ylim(bottom= -0.04, top = end-0.05)
            ax.set_xlabel('Energy')
            ax.set_ylabel('Entropy')
            ax.set_title(self.s_number + title)
            
            '''
            ax.text(0.5, 0.5, '$S_z$ = '+str(S_z_total_number),
            horizontalalignment='center',
            verticalalignment='center',
            transform = ax.transAxes)
            '''
            
            filename = 'entropy_'
            if suffix is not None:
                filename += suffix
            plt.savefig(os.path.join(self.directory, filename + '.png'), bbox_inches='tight', dpi=200)
            plt.close()
            
        def plot_lambdas_entropy(self, lambdas, color, title, figsize, s, suffix):
            # plotting -lambdy dla odpowiedniej entropii
            
            fig, ax = plt.subplots(figsize=figsize)
            for i in range(len(lambdas)):
                l1= ax.scatter([i]*len(lambdas[i]),lambdas[i], c = color , s=s , marker="_", linewidth=5, zorder=3)
                l2 =ax.scatter(i,sum(lambdas[i]), c = "green" , s=s , marker="_", alpha=.5, linewidth=4, zorder=3)
            
            ax.grid(axis='y')
            start, end = ax.get_ylim()
            ax.yaxis.set_ticks(np.arange(start, end, 0.05))
            ax.margins(0.1)
            ax.set_ylim(bottom=-0.04, top = 1.04)
            ax.set_xlabel('Number of eigenvector')
            ax.set_ylabel('energy of $\lambda$')
            ax.set_title(self.s_number + title)
            ax.legend((l1, l2), ('$\lambda$', 'Sum of $\lambda$'), loc='upper left', shadow=False)
            
            filename = 'lambda_entropy'
            if suffix is not None:
                filename += suffix
            plt.savefig(os.path.join(self.directory, filename + '.png'), bbox_inches='tight', dpi=200)
            plt.close()
            
        def plot_s_z(self,s_z_values, color, title, figsize, s, suffix):
            
            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(s_z_values, self.energies, c = color , s=s , marker="_", alpha = .8, linewidth=8, zorder=3)
            
            ax.grid(axis='y')
            ax.margins(0.1)
            ax.set_xlabel('$S_z$')
            ax.set_ylabel('$E$')
            ax.set_title(self.s_number + title)
            
            filename = 's_z_energy'
            if suffix is not None:
                filename += suffix
            plt.savefig(os.path.join(self.directory, filename + '.png'), bbox_inches='tight', dpi=200)
            plt.close()
 
def main(N, S, adjMatrix): #N+1 -> size of graph

    H = Heisenberg(N, S)
    
    #diagonalization of Heisenberg Hamiltonian 
    energies, vectors = H.diagonalize_Hamiltonian(adjMatrix)
    #calculation of S_z 
    S_z_total = H.calculate_S_z()
    
    print("Not rounded S_z: ", S_z_total)
    print("rounded S_z: ", np.around(S_z_total,0))
    print("sum rounded S_z: ", sum(np.around(S_z_total,0)))
    print("energy: ", energies[0])
    print("vector: ", vectors[:,0])
    print("vector rounded: ", np.around(vectors[:,0],3))
      
    Plotting = Plots(S, energies)
    #plots of energy bands 
    #H.plot_bands(title = "s=1, " + str(N+1) +" sites, graph", figsize=(10,12),s=100, ticks = False)
    Plotting.plot_bands(title = ", " +str(N+1) +" sites, graph", figsize=(10,12),s=550, ticks = True, suffix = str(N+1) +"_sites_chain")
    Plotting.plot_bands_with_s_z(np.around(S_z_total,0), title = ", " + str(N+1) +" sites, graph", figsize=(10,12),s=550, ticks = True, suffix = str(N+1) +"S_z_sites_chain")
    
    Plotting.plot_s_z(S_z_total, color = 'dodgerblue', title = ", " + str(N+1) +" sites, graph", figsize=(10,12),s=550, suffix = str(N+1) +"_sites_Sz")
   
    #basis = H.calculate_basis()
    #print("Our basis is: ", basis)

    entropy_all_system = []
    entropy_all_env = []
    
    eigen_rho_env_all = []
    eigen_rho_sys_all = []
    
    n_of_sites = int(len(adjMatrix)/2)
    
    for n in range(len(energies)):
        #print("This is " + str(n) + " eigenvector :", np.around(vectors[:,n],3))
        #print("This is " + str(n) + " S_z of the eigenvector :", np.around(S_z_total[n],1))
        
        rho_big= H.calculate_rho(n)
        
        rho_sys = H.calculate_reduced_rho_sys(rho_big, spin = S, sites_in_subsystem = n_of_sites)
        rho_env = H.calculate_reduced_rho_env(rho_big, spin = S, sites_in_subsystem = n_of_sites)
        
        #rho_2 = H.calculate_reduced_rho_2_spin(rho_sys)
        #rho_env = H.calculate_reduced_rho_2_spin_env(rho_sys)
        #rho_sys = H.calculate_reduced_rho_4_spin_sys(rho_big)
        #rho_env = H.calculate_reduced_rho_4_spin_env(rho_big)
        
        if not .9999999999 <= np.trace(rho_sys) <= 1.000000001:
            print("Trace of the system: ", np.trace(rho_sys))
            print("Trace of the env: ", np.trace(rho_env))
            
        entropy_sys, eigen_rho_sys = H.calculate_entropy(rho_sys, n_of_sites)
        entropy_all_system.append(entropy_sys)
        #for lambdas from reduced density matrices
        eigen_rho_sys_all.append(eigen_rho_sys)
        
        entropy_env, eigen_rho_env  = H.calculate_entropy(rho_env, n_of_sites)
        entropy_all_env.append(entropy_env)
        #for lambdas from reduced density matrices
        eigen_rho_env_all.append(eigen_rho_env)
    
    #print("List of entropies :", entropy_all_system)
    #print("Max entropy: ", max(entropy_all_env))
    #print("List of entropies rounded :", np.around(entropy_all_system,1))

    Plotting.plot_entropy(entropy_all_system, color = 'red', title = ", " + str(N+1) +" sites, system ", figsize=(10,12),s=550, suffix = str(N+1) + "_sys_entropy")
    Plotting.plot_entropy(entropy_all_env, color = 'blue', title = ", " + str(N+1) +" sites, environment ", figsize=(10,12),s=550, suffix = str(N+1) + "_env_entropy")
    
    Plotting.plot_lambdas_entropy(eigen_rho_env_all, color = 'black', title = ", lambda, " + str(N+1) +" sites, env ", figsize=(10,12),s=400, suffix = str(N+1) + "_env_lambda")
    Plotting.plot_lambdas_entropy(eigen_rho_sys_all, color = 'black', title = ", lambda, " + str(N+1) +" sites, sys ", figsize=(10,12),s=400, suffix = str(N+1) + "_sys_lambda")

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
    
    #in this code it's enough to define one "hopping", becasue the second one is already implemented in the code
    # correct above note! 
    #4 sites closed
    adjMatrix = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]])
    
    #4 sites open
    #adjMatrix = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]])
    
    #6 sites
    #adjMatrix = np.array([[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[1,0,0,0,0,0]])
    
    #8 sites
    #adjMatrix = np.eye(8, k=1, dtype=int)[::]
    #adjMatrix[-1][0] = 1
    
    #10 sites
    #adjMatrix = np.eye(10, k=1, dtype=int)[::]
    #adjMatrix[-1][0] = 1
    
    #12 sites
    #adjMatrix = np.eye(12, k=1, dtype=int)[::]
    #adjMatrix[-1][0] = 1
    
    S = 1/2
    main(len(adjMatrix) - 1, S, adjMatrix)
    print("Success")

