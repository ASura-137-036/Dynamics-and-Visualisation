import numpy as np
from scipy.integrate import solve_ivp,odeint
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import namedtuple
import vispy
import vispy
vispy.use('pyside6') #or pyqt5, pyqt6, pyside2, pyside6 if there is an error
from vispy import app, scene
from vispy.scene.visuals import create_visual_node
import imageio
from numpy import sin, cos

yt=np.load('double_pendulum.npz')['yt']
t=np.load('double_pendulum.npz')['t']

n_sys = yt.shape[1] 

cols = plt.cm.jet(np.random.rand(n_sys))#np.random.rand(n_sys,4)

print(cols.shape)


def pendulum_points(yt, i):
    pend_points = np.concatenate((np.zeros((n_sys, 1, 2)),
        (L1 * np.array([[sin(yt[i, :, 0])], [-cos(yt[i, :, 0])]])).transpose(2, 1, 0),
        (L2 * np.array([[sin(yt[i, :, 2])],[-cos(yt[i, :, 2])]]) + L1 * np.array([[sin(yt[i, :, 0])], [-cos(yt[i, :, 0])]])).transpose(2, 1, 0)),axis=1)
    return pend_points

#def pendulum_points(yt, i):
#    pend_points = np.zeros((n_sys, 1, 2))
#    pend_points = np.append(pend_points, (L1 * np.array([[sin(yt[i, :, 0])], [-cos(yt[i, :, 0])]])).transpose(2, 1, 0), axis=1)
#    pend_points = np.append(pend_points, (L2 * np.array([[sin(yt[i, :, 2])],[-cos(yt[i, :, 2])]]) + L1 * np.array([[sin(yt[i, :, 0])], [-cos(yt[i, :, 0])]])).transpose(2, 1, 0), axis=1)
#    return pend_points  

def pend_ss(yt, i ,l, wind=100):
    if i<=wind:
        ss=yt[0:i,:,2*(l-1):2*(l-1)+2].transpose(1,0,2)#(yt[0:i,:,2:4]).reshape(n_sys,i,-1, order= 'C')
    else:
        ss=yt[i-wind:i,:,2*(l-1):2*(l-1)+2].transpose(1,0,2)
    return ss

def pend_ssa(yt, i ,wind=100):
    if i<=wind:
        ss=yt[0:i,:,[1,3]].transpose(1,0,2)
    else:
        ss=yt[i-wind:i,:,[1,3]].transpose(1,0,2)
    return ss

def generate_colors(colors,M):
    #repeat same color for each vertices 
    #colors has length of number of segments
    #returns with colors of length of number of vertices
    if isinstance(M, np.ndarray) :
        n = M.shape[0]
        m = M.shape[1]
        #print(n,m)
        # make continuous connect for first segment
        out =(np.repeat(colors[:,np.newaxis], m, axis=1))
        #print(out.shape)
        return out.reshape(-1, colors.shape[1])
    elif isinstance(M, list):
        # if M is a list of arrays, we loop through each array repeat same color for each segment
        colout = []
        for i, Mi in enumerate(M):
            if isinstance(M, list):
                Mi = np.array(Mi)
            m = Mi.shape[0]
            #print(m)
            colout.append(np.repeat(colors[i][np.newaxis, :], m, axis=0))
        return np.concatenate(colout, axis=0)
    
def generate_connect(M):
    #if M is an array of shape (n, m,2/3) we use vectorisation to generate the connect array
    # else if M is a list of arrays we use a loop
    if isinstance(M, np.ndarray) and M.ndim == 3:
        n = M.shape[0]
        m = M.shape[1]

        # make continuous connect for first segment
        c1 = np.arange(0, m-1,dtype=int)
        c1 = np.column_stack((c1, c1 + 1))

        # repeat this for each segment and make a 2D array
        conn = np.repeat(c1[np.newaxis, :, :], n, axis=0)
        conn = conn + (np.arange(n) * m)[:,np.newaxis, np.newaxis]
        #flatten both connect and M
        conn = conn.reshape(-1, M.shape[2])
        M = M.reshape(-1, M.shape[2])
    elif isinstance(M, list):
        # if M is a list of arrays, we loop through each array and generate the connect array
        conn = []
        sc = 0
        for i, Mi in enumerate(M):
            if isinstance(M, list):
                Mi = np.array(Mi)
            m = Mi.shape[0]
            
            
            c1 = np.arange(0, m-1,dtype=int)
            #print(c1)
            c1 = np.column_stack((c1, c1 + 1))
            conn.append(c1 + sc)
            sc = sc + m
        conn = np.concatenate(conn, axis=0)
        M = np.concatenate(M, axis=0)
    else:
        raise ValueError("Input must be a 3D numpy array or a list of 2D arrays.")
    return conn,M

G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
L = L1 + L2  # maximal length of the combined pendulum
M1 = 2.0  # mass of pendulum 1 in kg
M2 = 5.0  # mass of pendulum 2 in kg
t_stop = 2.5  # how many seconds to simulate
history_len = 500  # how many trajectory points to display

import vispy.plot as vp

def black_plot(plot1):
    plot1.view.bgcolor = 'black'
    plot1.xaxis.axis.tick_color = 'white'
    plot1.yaxis.axis.tick_color = 'white'
    plot1.xaxis.axis.text_color = 'white'
    plot1.yaxis.axis.text_color = 'white'
    plot1.title._text_visual.color = 'white'
    return plot1

# Create a canvas showing plot data
fig = vp.Fig(bgcolor='black', size=(1200, 800))
pos= pendulum_points(yt, 450)
pos_ss = pend_ss(yt, 450,1)

colors = generate_colors(cols, pos)
connect,pos = generate_connect(pos)

colors_ss = generate_colors(cols, pos_ss)
connect_ss,pos_ss = generate_connect(pos_ss)

pos_ssa = pend_ssa(yt, 450)

colors_ssa = generate_colors(cols, pos_ssa)
connect_ssa,pos_ssa = generate_connect(pos_ssa)

pos_ss2 = pend_ss(yt, 450,2)
colors_ss2 = generate_colors(cols, pos_ss2)
connect_ss2,pos_ss2 = generate_connect(pos_ss2)

plot1 = fig[0, 0]
plot2 = fig[0, 1:3]
plot3= fig[1, 0]
plot4= fig[1, 1:3]

#plot
Line=plot1.plot(pos, connect=connect, color=colors, width=3, marker_size=0, title='Double Pendulum')
plot1.xaxis.axis.axis_label = 'X'
plot1.yaxis.axis.axis_label = 'Y'
Line_ss1=plot2.plot(pos_ss, connect=connect_ss, color=colors_ss, width=3, marker_size=0,
                    title='State Space of First link of Double Pendulum')
plot2.xaxis.axis.axis_label = 'theta_1'
plot2.yaxis.axis.axis_label = 'omega_1'
Line_ssa=plot3.plot(pos_ssa, connect=connect_ssa, color=colors_ssa, width=3, marker_size=0
                    ,title='Omega_1 vs Omega_2')
plot3.xaxis.axis.axis_label = 'omega_1'
plot3.yaxis.axis.axis_label = 'omega_2'
Line_ss2=plot4.plot(pos_ss2, connect=connect_ss2, color=colors_ss2, width=3, marker_size=0
                    ,title='State Space of Second Link of Double Pendulum')
plot4.xaxis.axis.axis_label = 'theta_2'
plot4.yaxis.axis.axis_label = 'omega_2'


plot1.view.camera.set_range(x=(-L, L), y=(-L, L))
plot1.view.camera.aspect = 1
plot2.view.camera.set_range(x=(-15, 15), y=(-15, 15))
#plot2.view.camera.aspect = 1
plot3.view.camera.set_range(x=(-15, 15), y=(-15, 15))
#plot3.view.camera.aspect = 1
plot4.view.camera.set_range(x=(-3, 27), y=(-15, 15))

black_plot(plot1)
black_plot(plot2)
black_plot(plot3)
black_plot(plot4)
i=0
colors_ss = generate_colors(cols, pos_ss)
colors = generate_colors(cols, pos)
colors_ssa = generate_colors(cols, pos_ssa)
colors_ss2 = generate_colors(cols, pos_ss2)
def update_plot(ev):
    global i, yt, L1, cols, colors_ss, colors, colors_ssa, colors_ss2
    wind=100
    i = (i + 1) % yt.shape[0]
    pos = pendulum_points(yt, i)
    pos_ss = pend_ss(yt, i,1)
    pos_ssa = pend_ssa(yt, i)
    pos_ss2 = pend_ss(yt, i,2)
    
    


    if True:#i <= wind + 2:
        colors_ss = generate_colors(cols, pos_ss)
        colors = generate_colors(cols, pos)
        colors_ssa = generate_colors(cols, pos_ssa)
        colors_ss2 = generate_colors(cols, pos_ss2)
    connect,pos = generate_connect(pos)
    connect_ss,pos_ss = generate_connect(pos_ss)
    connect_ssa,pos_ssa = generate_connect(pos_ssa)
    connect_ss2,pos_ss2 = generate_connect(pos_ss2)
    Line.set_data(pos, connect=connect, color=colors)
    Line_ss1.set_data(pos_ss, connect=connect_ss, color=colors_ss)
    Line_ssa.set_data(pos_ssa, connect=connect_ssa, color=colors_ssa)
    Line_ss2.set_data(pos_ss2, connect=connect_ss2, color=colors_ss2)
   #fig.scene.canvas.update()
    print(f"Progress: {i}/{yt.shape[0]}",end='\r')
    #if i > yt.shape[0] - 1:
    #    timer.stop()
    #    timer.disconnect()

#timer = app.Timer(interval=0.01, connect=update_plot)

def main():
    #fig.app.run()
    #timer.start()
    writer =imageio.get_writer("v_out.mp4",fps=40,
                           format='ffmpeg', ffmpeg_params=['-b:v', '4000k','-s', '1200x800'])

    for i in range(yt.shape[0]-2):#yt.shape[0]
        update_plot(None)
        print(f"Progress: {i}/{yt.shape[0]}", end='\r')
        frame=fig.scene.canvas.render()
        writer.append_data(frame)
        del frame
        fig.scene.canvas.update()
        fig.scene.canvas.app.process_events() 
    writer.close()

if __name__ == '__main__':
    main()
    