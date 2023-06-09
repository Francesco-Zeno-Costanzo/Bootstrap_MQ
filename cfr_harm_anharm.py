import time
import numpy as np
from scipy.sparse import diags
import  matplotlib.pyplot  as  plt
from scipy.integrate import simpson
from scipy.sparse.linalg import eigsh

start = time.time()

def U(x, g):
    '''
    Potential
    
    Parameter
    ---------
    x : 1darray or float
        array of position or position
    g : float
        coupling constant for quartic term
    
    Return
    ------
    x**2 + g*x**4 : 1darray or float
        potential
    '''
    return x**2 + g*x**4

 
def diag_H(n, h, x, f, m, args=()): 
    '''
    Comupte the lowest eigenvalues and relative eigenvectors
    
    Parameters
    ----------
    n : int
        dimension of matrix
    h : float
        step for derivative
    x : 1darray
        array of position
    f : callable
        function for potential
    m : int
        how many eigenvalues compute
    args : tuple
        extra args to pass to f
    
    Retunr
    ------
    avals : 1darray
        array of eigenvalue
    psi : 2darray
        wave function, psi[:,i] is associated to avals[i]
    '''
 
    P = diags([1, -2, 1], [-1, 0, 1], shape=(n, n))
    V = diags(f(x, *args), 0, shape=(n, n))
    H = -(1/(h**2))*P + V

    aval, avec = eigsh(H, k=m, which='SM')
    
    avals = np.sort(aval)
    avecs = avec[:,aval.argsort()]
    psi   = avecs/np.sqrt(h)
    
    return avals, psi


def E_per(g, m):
    '''
    Vaule of energy for anharmonic oscillator in
    pertubative theory at second order
    
    Parameter
    ---------
    g : float
        coupling constant for quartic term
    m : int
        energetic level
    
    Return
    ------
    E_n : float
        value of energy (x2 because above H = P^2 + V instead of P^2/2 + V)
    '''
    E_n = (m + 0.5) + g*(3/4*(m + 0.5)**2 + 3/16) - g**2 * (17/16*(m + 0.5)**3 + 67/64*(m + 0.5))
    return E_n*2


#===============================================================================
# Computational parameter
#===============================================================================
     
m  = 10                    # how many levels compute
n  = 1000                  # size of matrix
xr = 10                    # bound
xl = -10                   # bound
L  = xr - xl               # dimension of box
h  = (xr - xl)/(n)         # step size
tt = np.linspace(0, n, n)  # array form 0 to n
xp = xl + h*tt             # array of position

#===============================================================================
# Computation
#===============================================================================
    
eig_val_osc, psi_a = diag_H(n, h, xp, U, m, args=(0,)   )        # harmonic oscillator
eig_val_g_p, psi_p = diag_H(n, h, xp, U, m, args=(0.05,))        # anharmonic oscillator with g = 0.05
eig_val_g_1, psi_1 = diag_H(n, h, xp, U, m, args=(1,)   )        # anharmonic oscillator with g = 1
ene_g_p_p_2        = [E_per(0.05, i) for i in range(m)]          # energy with pertubation theory at second order with g = 0.05
ene_g_1_p_2        = [E_per(1, i) for i in range(m)]             # energy with pertubation theory at second order with g = 1

end = time.time()
print(f"Elapsed time: {end-start:.3f} s")

print("Tq & N x^2  & N gx^4 & P gx^4 & N 1x^4 & P 1x^4")
print("-----------------------------------------------")
for i in range(m):
    if 2*i + 1 < 10:
        if 2*i + 1 < 8:
            if 2*i + 1 < 6:
                print(f"{2*i+1}  & {eig_val_osc[i]:.3f}  & {eig_val_g_p[i]:.3f}  & {ene_g_p_p_2[i]:.3f}  & {eig_val_g_1[i]:.3f}  & {ene_g_1_p_2[i]:.3f} ")
            else:
                print(f"{2*i+1}  & {eig_val_osc[i]:.3f}  & {eig_val_g_p[i]:.3f}  & {ene_g_p_p_2[i]:.3f}  & {eig_val_g_1[i]:.3f} & {ene_g_1_p_2[i]:.3f} ")
        else:
            print(f"{2*i+1}  & {eig_val_osc[i]:.3f}  & {eig_val_g_p[i]:.3f} & {ene_g_p_p_2[i]:.3f} & {eig_val_g_1[i]:.3f} & {ene_g_1_p_2[i]:.3f}")
    else:
        print(f"{2*i+1} & {eig_val_osc[i]:.3f} & {eig_val_g_p[i]:.3f} & {ene_g_p_p_2[i]:.3f} & {eig_val_g_1[i]:.3f} & {ene_g_1_p_2[i]:.3f}")

x_n = np.array([simpson(abs(psi_1[:,j])**2 * xp**2, xp) for j in range(m)])
print(x_n)

#===============================================================================
# Save on File
#===============================================================================

"""
teo = np.array([2*i +1  for i in range(m)])

file = open("eig.txt", "w")
file.write("# teo \t eig_val_osc \t eig_val_g_p \t ene_g_p_p_2 \t eig_val_g_1 \t ene_g_1_p_2 \t x_n \n")

for a, b, c, d, e, f, g, in zip(teo, eig_val_osc, eig_val_g_p, ene_g_p_p_2, eig_val_g_1, ene_g_1_p_2, x_n):
    file.write(f"{a} \t {b} \t {c} \t {d} \t {e} \t {f} \t {g} \n")

file.close()
"""

#===============================================================================
# Plot
#===============================================================================

plt.figure(1)
plt.title("$\psi(x)$", fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('$\psi(x)$', fontsize=15)
plt.grid()
plt.ylim(0, 10)
plt.xlim(-5, 5)
ll = ['-', '--', '-.']

plt.plot(xp, U(xp, 0),    color='black', label='V(x)= $ x^2 $')
plt.plot(xp, U(xp, 0.05), color='red',   label='V(x)= $ x^2 + 0.05x^4$')
plt.plot(xp, U(xp, 1),    color='blue',  label='V(x)= $ x^2 + x^4$')
    
for L in range(3):

    plt.errorbar(xp, psi_a[:,L] + eig_val_osc[L], color='black',fmt=ll[L])
    plt.errorbar(xp, psi_p[:,L] + eig_val_g_p[L], color='red',  fmt=ll[L])
    plt.errorbar(xp, psi_1[:,L] + eig_val_g_1[L], color='blue', fmt=ll[L])

    plt.plot(xp, np.ones(len(xp))*eig_val_osc[L], color='black', linestyle='--', label='$E_{%d}=%f$' %(L, eig_val_osc[L]))
    plt.plot(xp, np.ones(len(xp))*eig_val_g_p[L], color='black', linestyle='--', label='$E_{%d}=%f$' %(L, eig_val_g_p[L]))
    plt.plot(xp, np.ones(len(xp))*eig_val_g_1[L], color='black', linestyle='--', label='$E_{%d}=%f$' %(L, eig_val_g_1[L]))

plt.legend(loc='best')


plt.show()
