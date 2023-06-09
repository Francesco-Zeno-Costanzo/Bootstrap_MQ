import time
import numpy as np
from scipy.integrate import simpson
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def boot(N, g, x2, Ene):
    '''
    bootstrap for anharmonic oscillators
    
    Parameters
    ----------
    N : int
        size of Hankel matrix
    g : float
        coupling constant for quartic term
    x2 : 1darray
        array of possible values of <x^2>
    Ene : 1darray
        array of possible values of energy
    
    Returns
    -------
    grid : 2darray
        grid of possible value of <x^2> and Ene:
        grid[i, j] = 1 if x[i] E[j] are god candidates
        grid[i, j] = 0 if x[i] E[j] are bad candidates
    '''
    steps = len(Ene)                    # steps**2 = number of points
    dim   = 2*N                         # how many momenta to calculate
    grid  = np.zeros((steps, steps))    # grid for result
    
    for l, x in enumerate(x2):
        for i, E in enumerate(Ene):     # loop over all possible points
        
            Mat_xm = np.zeros((N, N))   # initialize Hankel matrix
            moment = np.zeros(dim)      # initialize momenta array
            
            moment[0] = 1               # normalization
            moment[2] = x               # parameter
            moment[4] = (E - 2*x)/(3*g) # from ricorsion with m=1
            
            for k in range(3, dim-3, 2):# ricorsion relation
                moment[k+3] = (4*k*E*moment[k-1] + k*(k-1)*(k-2)*moment[k-3] - 4*(k+1)*moment[k+1])/(4*g*(k+2))
                # all odd momenta are zeros
                
            for h in range(N):          # create Hankel matrix
                for j in range(N):
                    Mat_xm[h, j] = moment[h+j]
            
            if(np.all(np.linalg.eigvals(Mat_xm) > 0)):
                # if Mat_xm is positive definite we have good candidates
                grid[l,i] = 1
                
    return grid


def value(grid, x2, Ene):
    '''
    Computation of mean value and associate error    
    
    Parameters
    ----------
    grid : 2darray
        result from boot is like the distribution
        for the two parameters
    x2 : 1darray
        array of possible values of <x^2>
    Ene : 1darray
        array of possible values of energy
        
    Results
    -------
    av_E, d_E : float
        <E> +- d_E
    av_X, d_X : float
        <x^2> +- d_x
    '''
    
    N = len(x2)
    
    P_E = np.array([simpson(grid[:, i], x2)  for i in range(N)])
    P_X = np.array([simpson(grid[i, :], Ene) for i in range(N)])
    
    N_E = simpson(P_E, Ene)
    N_X = simpson(P_X, x2)
    
    P_E = P_E/N_E
    P_X = P_X/N_X
    
    av_E = simpson(Ene * P_E, Ene)
    av_X = simpson(x2 * P_X, x2)
    d_E  = np.sqrt(simpson(Ene**2 * P_E, Ene) - av_E**2)
    d_X  = np.sqrt(simpson(x2**2 *  P_X, x2)  - av_X**2)
    
    print(f"average_E = {av_E:.5f} +- {d_E:.5f} & average_x2 = {av_X:.5f} +- {d_X:.5f}")
    
    return av_E, d_E, av_X, d_X

#===============================================================================
# Computational parameter
#===============================================================================

n = 0         # energetic level
g = 1         # coupling constant
steps = 400   # number of steps for good result 400 or higer

# value from code cfr_harm_anharm.py
teo, eig_val_osc, eig_val_g_p, ene_g_p_p_2, eig_val_g_1, ene_g_1_p_2, x_n = np.loadtxt("eig.txt", unpack=True)

x_min, x_max = 0.295, 0.312 # bound
E_min, E_max = 1.355, 1.425 # bound

x2   = np.linspace(x_min, x_max, steps) # all possible value
Ene  = np.linspace(E_min, E_max, steps) # all possible value

E, X = np.meshgrid(Ene, x2)      # for plot
grid = np.zeros((steps, steps))  # for plot

plt.figure(1)
plt.title(f"Anharmoic oscillator \n g={g}, n={n}", fontsize=15)
plt.xlabel("E", fontsize=15)
plt.ylabel(r"$\langle x^2 \rangle $", fontsize=15)

color = ["white", "blue", "red", "cyan", "yellow", "green"] 
cmap  = ListedColormap(color)

N_min = 8
N_max = 12 + 1 # N_max - N_min must be equal to len(color) - 1 for a good plot

#===============================================================================
# Start of computation
#===============================================================================

start = time.time()

for N in range(N_min, N_max):
    result = boot(N, g, x2, Ene) # Compute the grid
    value(result, x2, Ene)       # Compute vaule ad error form the region
    grid += result               # add to previus, for plot

end = time.time() - start
print(f"Elapsed time: {end} ")
    
#===============================================================================
# Plot
#===============================================================================

List = [f'k={i}' for i in range(N_min-1, N_max)]
List[0] = ''  # Label for colorbar

plt.axhline(y=x_n[n], label='from numerical simulation')
plt.axvline(x=eig_val_g_1[n]) 
   
c = plt.pcolor(E, X, grid, cmap=cmap)
cbar = plt.colorbar(c)

cbar.ax.get_yaxis().set_ticks([])
for j, lab in enumerate(List):
    cbar.ax.text(2.5, ((len(color)-1) * j + 2) / len(color), lab, ha='center', va='center')
cbar.ax.get_yaxis().labelpad = 15

plt.legend(loc='best')

plt.show()
