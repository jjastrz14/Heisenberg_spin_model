# version 3.0 for spin 1D chain diagonalization (max number of sites 8-10) using numpy and:
# -> calculating a density matrix
# -> calculating a reduced density matrix of chain divided into two equal subsytems
# -> calculating entropy of this subsystems
# -> making a matrix block diagonal by simple matrix operations
# -> calculation of rho reduced density matrices with fixed s_z quantum number
####


import numpy as np 
import Heisenberg_chain
import Plotting_writing
from itertools import groupby

        
if __name__ == '__main__':

    ######## Heisenberg Graph #########
    '''                                                                    
    Program for calculating and diagonalization of Heisenberg Hamiltonian for graph with defined adjacency matrix                               # 
    # class Graph - define graph, class Heisenberg - calculation of H
    # here you can creating a graph using add_edge methods or declare prepared matrix (adjMatrix).                                              #
    # N is a size of the system # 
    '''
    #sites = [4,6,8,10]
    #sites = [4,6,7,8,9]
    #for i in sites:
            #in this code it's enough to define one "hopping", becasue the second one is already implemented in the code
            #make above note more precise! 
            
            #4 sites open
            #adjMatrix = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]])

    size_of_the_chain = 12
    
    adjMatrix = np.eye(size_of_the_chain, k=1, dtype=int)[::]
    adjMatrix[-1][0] = 1
    #above matrix for size = 4 is : #adjMatrix = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]])
    print("This is adjacency Matrix : \n", adjMatrix)
    
    S = 1/2 #spin number 
    N = len(adjMatrix) #size of the system
    print("Calculating N = " + str(N)  + " system for S = " + str(S))
    
    H = Heisenberg_chain.Heisenberg(N, S)
    print("Start the diagonalization")
    
    basis_H, basis_H_s_z, spins = H.calculate_basis()

    #print("Basis H: ", basis_H)
    #print("Basis S_z: ", basis_H_s_z)
    print("List of possible values of S_z: ", spins)
    H.create_Hamiltonian(adjMatrix)
    
    all_energies = []
    entropy_all_system = []
    eigen_rho_sys_all = []
    s_z_number = []
    s_z_lambdas = []
    sum_lambdas = []
    
    #n_of_sites = int(len(adjMatrix)/2)
    size_of_sub_A = int(len(adjMatrix)/2)
    size_of_sub_B = N - size_of_sub_A
    
    for i in range(len(spins)):
        energies, vectors, spin_basis = H.block_Hamiltonian(i)
        
        print("S_z = " + str(spins[i]) + " start")
        #print(f"Spin basis: {spin_basis}")
        
        #calculation of new basis
        subsystem_A, subsystem_B, new_basis = H.subsystems_fixed_s_z(spin_basis,size_of_sub_A,size_of_sub_B)
        
        sum_entropy = 0
        for j in range(len(energies)):
            
            #print(f"Eigenvectors for this spin: {vectors[:,j]}")
            #print(set(vectors[:,j]))
            
            #if len(set(np.around(vectors[:,j],decimals = 5))) == 1:
                #print((f"Eigenvectors with equal elements: {vectors[:,j]}"))

            psi = np.zeros(shape=(len(subsystem_A),len(subsystem_B)), dtype = complex) #should deal only with floats
            for k,v in enumerate(vectors[:,j]):
                psi[new_basis[k][0]][new_basis[k][1]] = v #0 and 1 becasue it's a matrix 
            
            #print(f"This is psi vector {j}: \n", psi)
            #subsystemA
            rho = np.dot(psi, psi.conj().transpose()) 
            #print(f"This is rho vector {j}: \n", rho)
            
            #trace calculation
            if not .9999999999999 <= np.trace(rho) <= 1.000000000001:
                    print("Trace of the system: ", np.trace(rho))
            
            #entropy
            entropy_sys, eigen_rho_sys = H.calculate_entropy(rho, size_of_sub_A) #change to saving the lambdas
            print(f"This is entropy of energy {energies[j]} with spin S_z {spins[i]} : {entropy_sys}")
            sum_entropy += entropy_sys
            
            #[print(" lambdas: ", eigen_rho_sys[i]) for i  in range(len(eigen_rho_sys))]
            #[print("Complex lambdas: ", eigen_rho_sys[i]) for i  in range(len(eigen_rho_sys)) if eigen_rho_sys[i].imag <= 10e-8]
            
            entropy_all_system.append(entropy_sys)
            s_z_number.append(spins[i])
            
            s_z_lambdas.append([spins[i]]*len(eigen_rho_sys))
            #for lambdas from reduced density matrices
            eigen_rho_sys_all.append(eigen_rho_sys)
            
        #print(f"This is sum of this S_z {spins[i]} entropies {sum_entropy}")

        all_energies.append(energies)

        for i in range(len(eigen_rho_sys_all)):
                sum_lambdas.append(sum(eigen_rho_sys_all[i]))
        
            
        
    #print(sum_lambdas)
                
    all_energies = np.concatenate(all_energies)
    #energies_sorted = np.sort(all_energies)
    #print("Energies: ", energies_sorted)
    
    print("Start writing to files")
    if S == 1/2:
        S_save = "1_2"
    else:
        S_save = "1"
    
    #class initialization
    Plotting = Plotting_writing.Plots(S, directory = "./results/results_" + str(len(adjMatrix)) + "_sites_" + S_save)
        
    energies_entropies = [all_energies, entropy_all_system, s_z_number]
    Plotting.save_to_csv(energies_entropies, name = "/entropy_" + S_save + "_" + str(N), header = "Energies Entropies S_z", real = True)
    
    lambdas = np.column_stack([np.concatenate(eigen_rho_sys_all, axis = 0) , np.concatenate(s_z_lambdas)])
    Plotting.save_to_csv_without_transpose(lambdas, name = "/lambdas_" + S_save + "_" + str(N), header = "Lambdas S_z", real = True)
    
    Plotting.save_to_csv(sum_lambdas, name = "/sum_lambdas_" + S_save + "_" + str(N), header = "Sum", real = True)
    print("Writing to files done")
    
    
    #Plotting.plot_lambdas_entropy(eigen_rho_sys_all, color = 'black', title = ", lambda, " + str(N) +" sites, sys ", figsize=(10,12),s=400, suffix = str(N) + "_sys_lambda")
    #Plotting.plot_entropy(entropy_all_system, color = 'red', title = ", " + str(N) +" sites, system ", figsize=(10,12),s=550, suffix = str(N) + "_sys_entropy")
    
    print("Start plotting")
    
    if len(adjMatrix) <= 4: 
        ticks = True 
    else: ticks = False
    
    #plots of energy bands 
    Plotting.plot_bands(np.sort(all_energies), title = ", " +str(N) +" sites, graph", figsize=(10,12),s=550, ticks = ticks, suffix = str(N) +"_sites_chain")
    #Plotting.plot_bands_with_s_z(np.around(S_z_total,0), title = ", " + str(N+1) +" sites, graph", figsize=(10,12),s=550, ticks = True, suffix = str(N+1) +"S_z_sites_chain")
    Plotting.plot_s_z(sorted(basis_H_s_z), all_energies, color = 'dodgerblue', title = ", " + str(N) +" sites, graph", figsize=(10,12),s=550, suffix = str(N) +"_sites_Sz")
    Plotting.plot_entropy(all_energies, entropy_all_system, color = 'red', title = ", " + str(N) +" sites, system ", figsize=(10,12),s=550, suffix = str(N) + "_sys_entropy")
    #Plotting.plot_entropy(entropy_all_env, color = 'blue', title = ", " + str(N+1) +" sites, environment ", figsize=(10,12),s=550, suffix = str(N+1) + "_env_entropy")
    #Plotting.plot_lambdas_entropy(eigen_rho_env_all, color = 'black', title = ", lambda, " + str(N+1) +" sites, env ", figsize=(10,12),s=400, suffix = str(N+1) + "_env_lambda")
    Plotting.plot_lambdas_entropy(eigen_rho_sys_all, color = 'black', title = ", lambda, " + str(N) +" sites, sys ", figsize=(10,12),s=400, suffix = str(N) + "_sys_lambda")
    print("Plotting of bands done") 


    print("Success")





    
    
    
    

