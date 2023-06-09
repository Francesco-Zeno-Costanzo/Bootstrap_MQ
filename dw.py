import time
import numpy as np
from scipy.sparse import diags
import  matplotlib.pyplot  as  plt
from scipy.integrate import simpson
from scipy.sparse.linalg import eigsh

start = time.time()

def U(x, m2, g, v0):
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
    return -m2 * x**2 + g*x**4 + v0

 
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

m2 = 5#1
g  = 1#0.2
v0 = 25/4#1.25 

#===============================================================================
# Computation
#===============================================================================   

eigval, psi = diag_H(n, h, xp, U, m, args=(m2, g, v0))
x_n = np.array([simpson(abs(psi[:,j])**2 * xp**2, xp) for j in range(m)])

print("Ene    & <x^n>")
print("--------------")
for e, x in zip(eigval, x_n):
    if e < 10 :
        print(f"{e:.3f}  & {x:.3f}")
    else:
        print(f"{e:.3f} & {x:.3f}")

"""
file = open("eig_dw_2.txt", "w")
file.write("#eigval\t x_n \n")

for e, x in zip(eigval, x_n):
    file.write(f"{e} \t {x} \n")

file.close()
"""
end = time.time()
print(f"Elapsed time: {end-start:.3f} s")


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

plt.plot(xp, U(xp, m2, g, v0),    color='black', label=f'V(x)= $ -{m2}x^2 + {g}x^4 + {v0} $')
    
for L in range(3):

    plt.errorbar(xp, psi[:,L] + eigval[L], color='blue', fmt=ll[L])

    plt.plot(xp, np.ones(len(xp))*eigval[L], color='black', linestyle='--', label='$E_{%d}=%f$' %(L, eigval[L]))

plt.legend(loc='best')


plt.show()
