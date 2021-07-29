import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from scipy.integrate import odeint

Y0=[0.54,0.78,1.2,np.pi/4,0,0] #arreglo con las condiciones iniciales 
t=np.linspace(0,20,10000) #tiempo que dura la vaina (100!!!)

M=1
g=9.8
I=1
x0=2
y0=0
z0=0
c=M*g*x0

def g(Y,t):
    p,q,r,th,ph,ps = Y 
    
    dp = q*r/(2*I)
    dq = (-p*r - c*np.cos(th))/(2*I)
    dr = c*np.sin(th)*np.cos(ps)/I
    
    dth = p*np.cos(ps) - q*np.sin(ps)
    #dph = (1/(np.sin(th)*np.sin(ps)))*(p-dth*np.cos(ps))
    #dph = (1/(np.sin(th)*np.cos(ps)))*(q+dth*np.sin(ps))
    dph = (1/np.sin(th))*(p*np.sin(ps) + q*np.cos(ps))
    dps = r - dph*np.cos(th)
    
    return np.array([dp,dq,dr,dth,dph,dps])

sol=odeint(g,Y0,t)

w1 = sol[:,0]
w2 = sol[:,1]
w3 = sol[:,2]
theta = sol[:,3]
phi = sol[:,4]
psi = sol[:,5]

X = np.sin(psi)*np.sin(theta)
Y = np.cos(psi)*np.sin(theta)
Z = np.cos(theta)

fig = plt.figure(figsize=(14,9))
ax = p3.Axes3D(fig)

N=len(phi)

def update(num):
    x,y,z = X[num], Y[num], Z[num] 
    line.set_data(np.linspace(x0,x,100),np.linspace(0,y,100))
    line.set_3d_properties(np.linspace(0,z,100))

    trayectoria.set_data(X[0:num],Y[0:num])
    trayectoria.set_3d_properties(Z[0:num])
    return line, trayectoria

line, = ax.plot((x0,X[0]),(0,Y[0]),(0,Z[0]))
trayectoria, = ax.plot(X[0],Y[0],Z[0])

ani = animation.FuncAnimation(fig, update, N, interval=10000/N, blit=False)
plt.show()

ani.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
