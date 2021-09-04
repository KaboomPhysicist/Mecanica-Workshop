import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint


def solve_system(x0,y0,px0,py0,t0,tf): 
    Y0=[x0,y0,px0,py0] #arreglo con las condiciones iniciales 
    t=np.linspace(t0,tf,100000) #tiempo que dura la vaina

    l=1 #Parámetro lambda 

    def f(Y,t):
        x,y,px,py=Y 

        dx= px
        dpx=-x-2*l*x*y
        dy=py
        dpy=-y-l*(x**2-y**2)

        return ([dx,dy,dpx,dpy])

    sol=odeint(f,Y0,t)
    
    X=sol[:,0]
    Y=sol[:,1]
    PX=sol[:,2]
    PY=sol[:,3]
    
    return X,Y,PX,PY,t

fig = plt.figure(figsize=(14,9))
plt.title("Movimiento de la partícula")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")



sol0 = solve_system(0.1,0,0.1,0.02,0,150)
sol1 = solve_system(0.12,0,0.1,0.02,0,150)
sol2 = solve_system(0.14,0,0.1,0.02,0,150)
sol3 = solve_system(0.16,0,0.1,0.02,0,150)
sol4 = solve_system(0.18,0,0.1,0.02,0,150

N=len(sol0[0])

global vel
vel = 30

X0,Y0=sol0
X1,Y1=sol1
X2,Y2=sol2
X3,Y3=sol3
X4,Y4=sol4

plt.xlim(1.15*X0.min(),1.15*X0.max())
plt.ylim(1.15*Y0.min(),1.15*Y0.max())

plt.gca().set_aspect('equal','box')

def update(num):
    x,y = X[vel*num], Y[vel*num]

    masa.set_data(x,y)
    trayectoria1.set_data(X[max(0,int(0.05*vel*num)):vel*num],Y[max(0,int(0.05*vel*num)):vel*num])
    
    return  trayectoria1

masa0, = plt.plot(X0[0],Y0[0],'*',color='crimson',markersize=10)
masa1, = plt.plot(X1[0],Y1[0],'*',color='crimson',markersize=10)
masa2, = plt.plot(X2[0],Y2[0],'*',color='crimson',markersize=10)
masa3, = plt.plot(X3[0],Y3[0],'*',color='crimson',markersize=10)
masa4, = plt.plot(X4[0],Y4[0],'*',color='crimson',markersize=10)

trayectoria1, = plt.plot(X[0],Y[0],color='crimson')


ani = animation.FuncAnimation(fig, update, N//vel, interval=10000/(N//vel), blit=False)
plt.show()
#ani.save('ani001.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
