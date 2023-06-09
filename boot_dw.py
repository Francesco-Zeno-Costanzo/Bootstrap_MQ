import time
import numpy as np
from scipy.integrate import simpson
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def boot(N, x2, Ene, g, m2, v0):
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
            moment[4] = (4*E + 8*m2*x - 4*v0)/(12*g) # from ricorsion with m=1
            
            for k in range(3, dim-3, 2):# ricorsion relation
                moment[k+3] = (4*k*(E - v0)*moment[k-1] + k*(k-1)*(k-2)*moment[k-3] + 4*m2*(k+1)*moment[k+1])/(4*g*(k+2))
                # all odd momenta are zeros
                
            for h in range(N):          # create Hankel matrix
                for j in range(N):
                    Mat_xm[h, j] = moment[h+j]
            
            if(np.all(np.linalg.eigvals(Mat_xm) > 0)):
                # if Mat_xm is positive definite we have good candidates
                grid[l,i] = 1
                
                
    return grid


def value(grid, x2, Ene, n):
    '''
    Computation of mean value and associate error    
    
    Parameters
    ----------
    grid : 2darray
        result from boot
    
    x2 : 1darray
        array of possible values of <x^2>
    Ene : 1darray
        array of possible values of energy
    n : int
        number energetic level
        
    Results
    -------
    av_E, d_E : float
        <E> +- d_E
    av_X, d_X : float
        <x^2> +- d_x
    '''
    
    N = len(x2)
    M = len(Ene)
    
    P_E = np.array([simpson(grid[:, i], x2)  for i in range(M)])
    P_X = np.array([simpson(grid[i, :], Ene) for i in range(N)])
    
    N_E = simpson(P_E, Ene)
    N_X = simpson(P_X, x2)
    
    P_E = P_E/N_E
    P_X = P_X/N_X
    
    av_E = simpson(Ene * P_E, Ene)
    av_X = simpson(x2 * P_X, x2)
    d_E  = np.sqrt(simpson(Ene**2 * P_E, Ene) - av_E**2)
    d_X  = np.sqrt(simpson(x2**2 *  P_X, x2)  - av_X**2)
    
    print(f"average_E{n} = {av_E:.5f} +- {d_E:.5f} & average_x2 = {av_X:.5f} +- {d_X:.5f}")
    
    return av_E, d_E, av_X, d_X

#===============================================================================
# Computational parameter
#===============================================================================
p_set = 1

# Varius parameter of potential
g  = 0.2  if p_set == 1 else 1     # quartic coupling
m2 = 1    if p_set == 1 else 5     # square "mass"
v0 = 1.25 if p_set == 1 else 25/4  # shift

steps = 400   # number of steps for good result 400 or higer

# value from code dw.py
name_file = "eig_dw.txt" if p_set == 1 else "eig_dw_2.txt"
eigval, x_n = np.loadtxt(name_file, unpack=True)

x_min = 1.2 if p_set == 1 else 2.00 # different
x_max = 2.8 if p_set == 1 else 2.25 # bound
E_min = 0.7 if p_set == 1 else 2.80 # for each
E_max = 1.7 if p_set == 1 else 3.05 # case

x2   = np.linspace(x_min, x_max, steps) # all possible value
Ene  = np.linspace(E_min, E_max, steps) # all possible value

E, X = np.meshgrid(Ene, x2)      # for plot
grid = np.zeros((steps, steps))  # for plot

plt.figure(1)
plt.title(f"double well \n g={g}, $m^2$={m2} $v_0$={v0}", fontsize=15)
plt.xlabel("E", fontsize=15)
plt.ylabel(r"$\langle x^2 \rangle $", fontsize=15)

if p_set == 1:
    color = ["white", "blue", "red", "black", "yellow", "green", 'pink']
    dim_MH = [10, 11, 12, 13, 14, 15]
if p_set == 2:
    color = ["white", "blue", "red", "black", "yellow", "green", "pink"]
    dim_MH = [17, 18, 19, 18, 19, 20] 

cmap  = ListedColormap(color)

#===============================================================================
# Start of computation
#===============================================================================

start = time.time()

for N in dim_MH:
    result = boot(N, x2, Ene, g, m2, v0)       # compute the grid
    
    e_0, de0, _, _ = value(result[:, :steps//2 ], x2, Ene[:steps//2 ], 0)   # we search for two value of
    e_1, de1, _, _ = value(result[:,  steps//2:], x2, Ene[ steps//2:], 1)   # energy, so we split the grid
    
    grid += result    # add to previus, for plot
    
DE_b = e_1 - e_0
dDEb = np.sqrt( (de1/e_1)**2 + (de0/e_0)**2 ) * DE_b

DE_n = eigval[1] - eigval[0]

wkb = 8 * 2**(1/4) * m2**5 /np.sqrt(np.pi*g) * np.exp(-np.sqrt(2)*np.sqrt(m2)**3 /(3*g))

print(f'Splitting with bootstrap          = {DE_b:.5f} +- {dDEb:.5f}')
print(f'Splitting with numerical solution = {DE_n:.5f}') 
print(f'Splitting with wkb                = {wkb:.5f}')

end = time.time() - start
print(f"Elapsed time: {end} ")

#===============================================================================
# Plot
#===============================================================================

List = [f'k={i}' for i in dim_MH]
List.insert(0, '')  # Label for colorbar

plt.plot(eigval[0], x_n[0], '*')
plt.plot(eigval[1], x_n[1], '*',label='from numerical simulation')
 
c = plt.pcolor(E, X, grid, cmap=cmap)

cbar = plt.colorbar(c)

cbar.ax.get_yaxis().set_ticks([])
for j, lab in enumerate(List):
    cbar.ax.text(2.5, ((len(color)-1) * j + 2) / len(color), lab, ha='center', va='center')
cbar.ax.get_yaxis().labelpad = 15

plt.legend(loc='best')

plt.show()
