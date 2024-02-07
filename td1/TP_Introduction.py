#==============================================================================
# Processus stochastiques et EDPs 2023-2024
# Introduction au lien entre Probabilités et EDPs
# avec le problème de Dirichlet discrétisé sur un carré
# Attention à exécuter cellule par cellule
#==============================================================================

from math import pi
import numpy as np
from matplotlib import pyplot as plt

# Exemples de fonctions solution du problème (fonctions harmoniques)
# Température exacte connue de manière analytique  

def thetaExact1(x, y):
    return 20*x*y

def thetaExact2(x, y):
    return 20*np.cos(2*pi*x)*np.exp(2*pi*(y-1))

#%%   
# Méthode des différences finies
# Paramètres de discrétisation 
N = 32
h = 1/N

# Solution exacte sur les points de la grille de discrétisation
theta = np.zeros(shape=(N+1,N+1))
for i in range(N+1):
    for j in range(N+1):
        theta[i,j] = thetaExact1(h*i, h*j)
         
# Assemblage de la matrice
d = (N-1)**2
A = np.zeros(shape=(d,d))
for i in range(d):
    A[i,i] = 4
for i in range(d-1):
    A[i,i+1] = -1
    A[i+1,i] = -1
for i in range(d-(N-1)):
    A[i,i+N-1] = -1
    A[i+N-1,i] = -1
for i in range(N-2):
    A[i*(N-1)+N-2,i*(N-1)+N-1] = 0
for i in range(N-2):
    A[i*(N-1)+N-1,i*(N-1)+N-2] = 0

# Assemblage du second membre
b = np.zeros(shape=(d,1))

for i in range(N-1):
    b[i*(N-1)] = theta[1+i,0]
    b[i*(N-1)+N-2] = theta[1+i,N]

for j in range(N-1):
    b[j] += theta[0,1+j]
    b[(N-2)*(N-1)+j] += theta[N,1+j]

# Solution du système linéaire
thetaDF1D = np.linalg.solve(A,b)

# Formatage
thetaDF2D = np.zeros(shape=(N+1,N+1))

for i in range(N+1):
    thetaDF2D[i,0] = theta[i,0]
    thetaDF2D[i,N] = theta[i,N]
for j in range(1,N+1):
    thetaDF2D[0,j] = theta[0,j]
    thetaDF2D[N,j] = theta[N,j]

for i in range(1,N):
    for j in range(1,N):
        thetaDF2D[i,j] = thetaDF1D[(i-1)*(N-1)+j-1]

# print(thetaDF2D)

# Visualisation de la solution par différences finies
xx = np.linspace(0,1,N+1)
yy = np.linspace(0,1,N+1)
plt.figure()
plt.pcolormesh(xx,yy,thetaDF2D.T, shading='auto', cmap='jet') 
plt.axis("image")
plt.show()

# Valeur obtenue au milieu du carré (x=0.5, y=0.5)
print("\n","Valeur exacte :", thetaExact1(0.5,0.5),"\n",
      "Valeur approchée par DF :", thetaDF2D[(N//2),(N//2)])

#%%
# Méthode Monte-Carlo au milieu du carré

plt.close()

# Générateur de marche aléatoire sur un réseau (Random Walk)
def RW(position, pas):
    direction = np.random.randint(low=1,high=5,size=1)
    if direction==1:
        step = pas*np.array([1.,0])
    elif direction==2:
        step = pas*np.array([0,1.])
    elif direction==3:
        step = pas*np.array([-1.,0])
    else: step = pas*np.array([0,-1.])
    return position + step

pointCourant = np.array([0.5,0.5])
RWtheta = [thetaExact1(pointCourant[0],pointCourant[1])]

plt.figure()
plt.ion()
plt.pcolormesh(xx,yy,thetaDF2D.T, shading='auto', cmap='jet')
plt.axis('image')
plt.show()
plt.scatter(pointCourant[0], pointCourant[1], color='r')
plt.pause(5)          
           
pointSuivant = RW(pointCourant, h)
RWtheta.append(thetaExact1(pointSuivant[0],pointSuivant[1]))

xpoints = [pointCourant[0], pointSuivant[0]]
ypoints = [pointCourant[1], pointSuivant[1]]
plt.plot(xpoints, ypoints, 'ko-', markersize=1)

pointCourant, pointSuivant = pointSuivant, RW(pointSuivant, h)

compteur = 0
while (0 < pointSuivant[0] < 1) & (0 < pointSuivant[1] < 1):
    compteur += 1
    xpoints = [pointCourant[0], pointSuivant[0]]
    ypoints = [pointCourant[1], pointSuivant[1]]
    plt.plot(xpoints, ypoints, 'ko-', markersize=1)
    pointCourant, pointSuivant = pointSuivant, RW(pointSuivant, h)
    RWtheta.append(thetaExact1(pointSuivant[0],pointSuivant[1]))
    if compteur < 20:
        plt.draw()
        plt.pause(1)

  
plt.show()
xpoints = [pointCourant[0], pointSuivant[0]]
ypoints = [pointCourant[1], pointSuivant[1]]
plt.plot(xpoints, ypoints, 'k-')
plt.scatter(pointSuivant[0], pointSuivant[1],color='r')

plt.pause(5)
plt.figure()
plt.plot(RWtheta,color='r',linewidth=2)
plt.xlabel(u'$Temps$', fontsize=10)
plt.ylabel(u'$Température$', fontsize=10)
plt.axis([0,len(RWtheta)-1,0,20])
plt.grid()
plt.title('Température le long de la marche aléatoire')
plt.show()

#%%
# Vérification de la formule de représentation probabiliste de la solution
# au milieu du carré

# ... à compléter

NMC = 200000
thetaMC = []

for nMC in range(NMC):
    pointCourant = np.array([0.5,0.5])
    pointSuivant = RW(pointCourant, h)
    pointCourant, pointSuivant = pointSuivant, RW(pointSuivant, h)
    while (0 < pointSuivant[0] < 1) & (0 < pointSuivant[1] < 1):
        xpoints = [pointCourant[0], pointSuivant[0]]
        ypoints = [pointCourant[1], pointSuivant[1]]
        pointCourant, pointSuivant = pointSuivant, RW(pointSuivant, h)
    thetaMC.append(thetaExact1(pointSuivant[0],pointSuivant[1]))
    
thetaHat = np.mean(thetaMC)

print("\n","Valeur exacte :", thetaExact1(0.5,0.5))
print(" Valeur approchée par DF :", thetaDF2D[(N//2),(N//2)])
print(" Valeur par MC avec ", NMC, " simulations : ", "%1.2f" %thetaHat)

se = np.std(thetaMC)/np.sqrt(NMC)
print(" IC à 95% : [", "%1.2f" %(thetaHat-2*se), ";", "%1.2f" %(thetaHat+2*se), "]")