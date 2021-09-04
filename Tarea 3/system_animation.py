import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint


def solve_system(x0,y0,px0,py0,t0,tf): 
    Y0=[x0,y0,px0,py0] #arreglo con las condiciones iniciales 
    t=np.linspace(t0,tf,1000000) #tiempo que dura la vaina

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


sol=solve_system(0,0.1,0.44,0.1,0,500)
N=len(sol[0])

global vel
vel = 10

X=sol[0]
Y=sol[1]
def update(num):
    x,y = X[vel*num], Y[vel*num]

    trayectoria1.set_data(X[max(0,int(0.1*vel*num)):vel*num],Y[max(0,int(0.1*vel*num)):vel*num])
    
    return  trayectoria1

trayectoria1, = plt.plot(X[0],Y[0])


ani = animation.FuncAnimation(fig, update, N//vel, interval=10000/(N//vel), blit=False)
plt.show()
ani.save('ani004.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
