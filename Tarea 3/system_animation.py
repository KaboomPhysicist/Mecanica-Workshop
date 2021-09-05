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
plt.title("Movimiento de la partícula")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")



sol0 = solve_system(0.1,0,0.03,0.5,0,150)
sol1 = solve_system(0.1,0,0.04,0.5,0,150)
sol2 = solve_system(0.1,0,0.05,0.5,0,150)
sol3 = solve_system(0.1,0,0.06,0.5,0,150)
sol4 = solve_system(0.1,0,0.07,0.5,0,150)

N = len(sol0[0])

global vel
vel = 300

X0,Y0=sol0[:2]
X1,Y1=sol1[:2]
X2,Y2=sol2[:2]
X3,Y3=sol3[:2]
X4,Y4=sol4[:2]

limin=1.15*np.min([*X0,*X1,*X2,*X3,*X4,*Y0,*Y1,*Y2,*Y3,*Y4])
limax=1.15*np.max([*X0,*X1,*X2,*X3,*X4,*Y0,*Y1,*Y2,*Y3,*Y4])


plt.xlim(limin,limax)
plt.ylim(limin,limax)

plt.gca().set_aspect('equal','box')

def update(num):
    x0,y0 = X0[vel*num], Y0[vel*num]
    x1,y1 = X1[vel*num], Y1[vel*num]
    x2,y2 = X2[vel*num], Y2[vel*num]
    x3,y3 = X3[vel*num], Y3[vel*num]
    x4,y4 = X4[vel*num], Y4[vel*num]


    masa0.set_data(x0,y0)
    masa1.set_data(x1,y1)
    masa2.set_data(x2,y2)
    masa3.set_data(x3,y3)
    masa4.set_data(x4,y4)

    trayectoria0.set_data(X0[max(0,int(0.05*vel*num)):vel*num],Y0[max(0,int(0.05*vel*num)):vel*num])
    trayectoria1.set_data(X1[max(0,int(0.05*vel*num)):vel*num],Y1[max(0,int(0.05*vel*num)):vel*num])
    trayectoria2.set_data(X2[max(0,int(0.05*vel*num)):vel*num],Y2[max(0,int(0.05*vel*num)):vel*num])
    trayectoria3.set_data(X3[max(0,int(0.05*vel*num)):vel*num],Y3[max(0,int(0.05*vel*num)):vel*num])
    trayectoria4.set_data(X4[max(0,int(0.05*vel*num)):vel*num],Y4[max(0,int(0.05*vel*num)):vel*num])
    

masa0, = plt.plot(X0[0],Y0[0],'*',color='crimson',markersize=10)
masa1, = plt.plot(X1[0],Y1[0],'*',color='purple',markersize=10)
masa2, = plt.plot(X2[0],Y2[0],'*',color='navy',markersize=10)
masa3, = plt.plot(X3[0],Y3[0],'*',color='cyan',markersize=10)
masa4, = plt.plot(X4[0],Y4[0],'*',color='teal',markersize=10)

trayectoria0, = plt.plot(X0[0],Y0[0],color='crimson')
trayectoria1, = plt.plot(X1[0],Y1[0],color='purple')
trayectoria2, = plt.plot(X2[0],Y2[0],color='navy')
trayectoria3, = plt.plot(X3[0],Y3[0],color='cyan')
trayectoria4, = plt.plot(X4[0],Y4[0],color='teal')



ani = animation.FuncAnimation(fig, update, N//vel, interval=10000/(N//vel), blit=False)
plt.show()
ani.save('ani002.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

