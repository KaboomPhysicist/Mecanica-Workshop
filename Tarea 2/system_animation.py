import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from scipy.integrate import odeint

def solve_system(p0, q0, r0, th0, ps0, t0, t1,nt=100000):
    #0.54,0.78,1.2,np.pi/4,0,0
    Y0=[p0, q0, r0, th0, ps0] #arreglo con las condiciones iniciales 
    t=np.linspace(t0,t1,nt) #tiempo que dura la vaina (100!!!)

    global M
    M=1
    g=-9.8
    I=1
    global x0, y0, z0
    x0=-2
    y0=0
    z0=0
    c=M*g*x0

    def g(Y,t):
        p,q,r,th,ps = Y 

        dp = q*r/(2*I)
        dq = (-p*r - c*np.cos(th))/(2*I)
        dr = c*np.sin(th)*np.cos(ps)/I

        dth = p*np.cos(ps) - q*np.sin(ps)
        dps = r - (1/np.sin(th))*(p*np.sin(ps) + q*np.cos(ps))*np.cos(th)

        return np.array([dp,dq,dr,dth,dps])
    
    sol=odeint(g,Y0,t)

    w1 = sol[:,0]
    w2 = sol[:,1]
    w3 = sol[:,2]
    theta = sol[:,3]
    psi = sol[:,4]
    
    return w1, w2, w3, theta, psi

def position(psi, theta):
    X = np.sin(psi)*np.sin(theta)
    Y = np.cos(psi)*np.sin(theta)
    Z = np.cos(theta)
    return X, Y, Z

w1, w2, w3, theta, psi = solve_system(1,0,1,np.pi/4,np.pi/8,0,10,10000)


r=2

#Posición del centro de masa. Está a r/4 desde el plano del anillo
Z = x0*np.sin(psi)*np.sin(theta)
Y = x0*np.cos(psi)*np.sin(theta)
X = x0*np.cos(theta)

func_x1 = lambda y: (1/(x0**2+z0**2))*(x0**3-x0*y0*y+x0*y0**2+x0*z0**2+np.sqrt(-z0**2*((y-y0)**2*(x0**2+y0**2+z0**2)-r**2*(x0**2+z0**2))))
func_x2 = lambda y: (1/(x0**2+z0**2))*(x0**3-x0*y0*y+x0*y0**2+x0*z0**2-np.sqrt(-z0**2*((y-y0)**2*(x0**2+y0**2+z0**2)-r**2*(x0**2+z0**2))))

func_z1 = lambda x,y: 0.5*(z0+np.sqrt(z0**2+4*(r**2-x**2-y**2+x*x0+y*y0)))
func_z2 = lambda x,y: 0.5*(z0-np.sqrt(z0**2+4*(r**2-x**2-y**2+x*x0+y*y0)))


def func_anillo(x0, y0, z0):

    Y_init = np.linspace(y0-5*r,y0+5*r,1000000)

    disc1 = lambda y: -(z0**2)*((y-y0)**2*(x0**2+y0**2+z0**2)-r**2*(x0**2+z0**2))

    n=0

    for pos, element  in enumerate(Y_init):
        if disc1(element)<0:
            n+=1
            Y_init = np.delete(Y_init, pos)
    print(n, len(Y_init))

    Y_duo = np.concatenate((Y_init, Y_init))

    func_x1 = lambda y: (1/(x0**2+z0**2))*(x0**3-x0*y0*y+x0*y0**2+x0*z0**2+np.sqrt(disc1(y)))
    func_x2 = lambda y: (1/(x0**2+z0**2))*(x0**3-x0*y0*y+x0*y0**2+x0*z0**2-np.sqrt(disc1(y)))

    func_z1 = lambda x,y: 0.5*(z0+np.sqrt(z0**2+4*(r**2-x**2-y**2+x*x0+y*y0)))
    func_z2 = lambda x,y: 0.5*(z0-np.sqrt(z0**2+4*(r**2-x**2-y**2+x*x0+y*y0)))

    func_x1(Y_init)
    func_x2(Y_init)

    return X,Y,Z

#func_anillo(2,0,3)

fig = plt.figure(figsize=(14,9))
ax = p3.Axes3D(fig)

N=len(psi)

global vel
vel = 10

def update(num):
    x,y,z = X[vel*num], Y[vel*num], Z[vel*num] 
    eje1.set_data(np.linspace(0,x,100),np.linspace(0,y,100))
    eje1.set_3d_properties(np.linspace(0,z,100))

    trayectoria1.set_data(X[max(0,int(0.1*vel*num)):vel*num],Y[max(0,int(0.1*vel*num)):vel*num])
    trayectoria1.set_3d_properties(Z[max(0,int(0.1*vel*num)):vel*num])
    return eje1, trayectoria1

eje1, = ax.plot((0,X[0]),(0,Y[0]),(0,Z[0]))
trayectoria1, = ax.plot(X[0],Y[0],Z[0])

ax.set_xlim3d([-2, 2.0])
ax.set_xlabel('Z')

ax.set_ylim3d([-2, 2.0])
ax.set_ylabel('Y')

ax.set_zlim3d([-5.0, 5.0])
ax.set_zlabel('X')


ani = animation.FuncAnimation(fig, update, N//vel, interval=10000/(N//vel), blit=False)
plt.show()
ani.save('ani004.mp4', fps=30, extra_args=['-vcodec', 'libx264'])