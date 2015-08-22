import numpy as np
import scipy.stats as st
import scipy.optimize as O
import Quandl


# Let x be a vector of time series points
# Let N be length(x)
# Let t = 1:N
# We want the likelihood ratio test for whether x comes from P versus Q
tok = "CoHxdmas1Pz2t7Hk8maE"
x = Quandl.get("WIKI/AAN",ahtoken=tok)
print(x)
r = np.diff(np.log(x["Adj. Close"]))
r[np.isnan(r)]=0





def spacebin( x0, xvals):
    c = 1000
    idx = 0
    for j in range(len(xvals)):
        if np.abs(x0-xvals[j])<c:
            idx = j
            c=np.abs(x0-xvals[j])
    return idx

def create2Ddensity(u, F):
   M,T = u.shape
   D = np.zeros((M,T))
   for i in range(M):
       for j in range(T):
           D[i,j] = F(u[i,j])
   return(D)
 
def density2Dconv( p, D, t0,dt,x0, dx):
    M,T = D.shape
    q = np.zeros((M,T))
    for i in range(M):
        for j in range(T):
            q[i,j] = 0
            for l in range(i):
                for m in range(M):
                    q[i,j] += dt*dx*D[l,m]*p[i-l,j-m]
    return(q)
    

def nlpart(p, a, alpha):
    q = p*0
    k = 0
    p0 = p**(alpha*k)
    for a0 in a:
        q += a0 * p0
        p0 = p0*p**alpha
    return(q)

def createF( alpha, a):
    return lambda(x): a[0]+a[1]*x**alpha + a[2]*x**(2*alpha) + a[3]*x**(3*alpha)
 
def normalize(p):
    return p/sum(p)

def nlevyllk(theta,data=r ): 
    fexp = theta[0]
    df = theta[1]

    a0 = theta[2]
    a1 = theta[3]
    a2 = theta[4]
    a3 = theta[5]

    alpha = (0.5+df)**(-1)
    F = createF( alpha, [a0,a1,a2,a3])
    t0=-2.00
    dt=0.4
    t=np.arange(t0,-t0,dt)
    f=np.exp( -fexp*t**2)
    g=np.fft.fft(f)
#frequency normalization factor is 2*np.pi/dt
    w = np.fft.fftfreq(f.size)*2*np.pi/dt
#In order to get a discretisation of the continuous Fourier transform
#we need to multiply g by a phase factor
    g*=dt*np.exp(-complex(0,1)*w*t0)/(np.sqrt(2*np.pi))
    N=len(r)

    z0 = (- (fexp/np.pi**2)*np.log(np.sqrt(fexp/np.pi)*g))
    z = np.exp(-fexp*abs(w*w)**(alpha))
    q = np.fft.ifft(z)
    q = np.real(q)/sum(abs(q))
    
    M = len(q)
    Q = np.zeros((len(t),len(r)))
    for j in range(len(r)):
        Q[:,j] = q
    D = create2Ddensity(Q,F)
        
    R = density2Dconv(Q,D,0.0,1.0,-t0,dt)
    P = Q + R

    # normalize each time point
    for j in range(len(r)):
        P[:,j] = P[:,j]/sum(P[:,j])

    llk = 0
    for j in range(N):
        idx = spacebin(r[j],t)
        llk = llk + np.log( P[idx,j])
    return(-llk)

llk1 =nlevyllk( [5.0,0.1,0.1,0.0,0.0,0.0],data=r)
llk2 =nlevyllk( [5.0,0.1,0.1,0.1,0.2,0.1],data=r)

print(llk1)
print(llk2)
print(2*llk1-2*llk2)

print(nlevyllk( [10.0,0.05,0.0,0.0,0.0,0.0],data=r))
theta0 = [4.0,0.05,0.0,0.0,0.0,0.0]
res = O.minimize( nlevyllk, theta0, method="Nelder-Mead" )
print(res.x)
print(res.fun) 

