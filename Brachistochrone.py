#https://github.com/MilesCranmer/PySR
#https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
#https://physics.nist.gov/cgi-bin/cuu/Value?gn
#https://homework.study.com/explanation/a-box-is-sent-up-a-frictionless-inclined-34-2-degrees-plane-at-an-initial-speed-of-3-9-m-s-a-how-much-time-in-seconds-does-it-take-for-the-box-to-stop-b-how-far-did-the-box-travel-up-the-p.html

#https://duckduckgo.com/?q=keras-rl+how+to+define+a+custom+environment&kp=1&t=h_&iax=videos&ia=videos&iai=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DbD6V3rcr_54
#https://stackoverflow.com/questions/58964267/how-to-create-an-openai-gym-observation-space-with-multiple-features

from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
ln, = ax.plot([], [])

x_i, y_i = (1,10)
x_f, y_f = (10,0)
times = []
x_coords = np.logspace(0, 1, 50)
best_y_coords = []

def init():
    ax.set_ylim(-30, 15)
    ax.set_xlim(1,10)
    return ln,

def update(frame):
    global best_y_coords
    g = 9.80665
    v = 0 #initial velocity
    t = 0 #total time
    y_coords = [y_i]

    y_coord = y_i
    x_coord = x_i
    y_coord_new = 0

    for x in x_coords[1:]:
        y_min = y_f - (v**2) / (2*g)
        y_max = (v**2) / (2*g) + y_coord
        if x == x_coords[-1]:
            y_coord_new = y_f
            assert(y_min <= y_coord_new <= y_max)
        else:
            y_coord_new = np.random.uniform(y_min,y_max)
            assert(y_min <= y_coord_new <= y_max)
        opp = abs(y_coord_new - y_coord)
        adj = abs(x-x_coord)
        theta = np.arctan(opp/adj)
        
        d = np.sqrt(opp**2 + adj**2)
        
        if y_coord_new > y_coord:
            v_f = np.sqrt(np.abs(v**2 - 2*g*d*np.sin(theta)))
            t += abs(v_f - v)/(g*np.sin(theta))
            v = v_f
        else:
            v_f = np.sqrt(np.abs(v**2 + 2*g*d*np.sin(theta)))
            t += abs(v_f - v)/(g*np.sin(theta))
            v = v_f
            
        y_coord = y_coord_new
        x_coord = x
        y_coords.append(y_coord)
    
    if not times:
        best_y_coords = y_coords
        times.append(t)
#        new_best = True
    elif t < min(times):
        print(t)
        best_y_coords = y_coords
        times.append(t)
#        new_best = True

    ln.set_data(x_coords, best_y_coords)
    return ln,
    
ani = FuncAnimation(fig, update, frames=np.linspace(0,1,100), init_func=init, blit=True, interval = 100)
plt.show()

print(min(times))
