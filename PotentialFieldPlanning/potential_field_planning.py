"""
Potential Field based path planner
author: Atsushi Sakai (@Atsushi_twi)
Ref:
https://www.cs.cmu.edu/~motionplanning/lecture/Chap4-Potential-Field_howie.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import math

# Parameters
show_animation = True

# Potential field parameters
f_ct = 1.0  # attractive potential gain
f_cr = 5 # repulsive potential gain
n = 2 # chosen dimension
k = 0.3 # proportional constant for steering
active_wdw_size = 4 # acitve window size [m] 
AREA_WIDTH = 30.0  # area width [m]

# Grid parameters
minx = - AREA_WIDTH / 2.0
miny = - AREA_WIDTH / 2.0
maxx = + AREA_WIDTH / 2.0
maxy = + AREA_WIDTH / 2.0


# Others parameters
omega_max = 0.8 #[s**(-1)]
v_max = 0.8 #[m.s^(-1)]


# Compute the potential for the given robot position
def calc_potential_field(x, y, theta, gx, gy, ox, oy, reso, rr):
    xw = int(round((maxx - minx) / reso))
    yw = int(round((maxy - miny) / reso))

    # calc each potential
    pmap = [[0.0 for i in range(yw)] for i in range(xw)]
    
    # sum of the two below defined potentials (attractive and repulsive)
    pmap, ft = calc_attractive_potential(pmap, x, y, gx, gy)
    pmap, fr, c1, c2, c3, c4 = calc_repulsive_potential(pmap, x, y,theta, ox, oy, rr)
    return pmap, fr+ft, c1, c2, c3, c4

# Attractive potential
def calc_attractive_potential(pmap, x, y, gx, gy):
    d = np.sqrt((x-gx)**2 + (y-gy)**2)
    fx = f_ct * (gx-x) / d
    fy = f_ct * (gy-y) / d
    pmap[int(gx)][int(gy)] = np.sqrt(fx**2+ fy**2)
    return pmap, np.array([fx,fy])
 
# Repulsive potential
def calc_repulsive_potential(pmap, x, y,theta, ox, oy, rr):
    fx = 0
    fy = 0
    # Calculate the coordinates of active window corners
    c1 = np.array([x+rr*np.cos(theta) + active_wdw_size*np.sin(theta),
                   y+rr*np.sin(theta) - active_wdw_size*np.cos(theta)])
    c2 = np.array([x+rr*np.cos(theta) - active_wdw_size*np.sin(theta),
                   y+rr*np.sin(theta) + active_wdw_size*np.cos(theta)])
    c3 = np.array([x+rr*np.cos(theta) + active_wdw_size*(-np.sin(theta)+np.cos(theta)),   y+rr*np.sin(theta) + active_wdw_size*(np.cos(theta)+np.sin(theta))])
    c4 = np.array([x+rr*np.cos(theta) + active_wdw_size*(np.sin(theta)+np.cos(theta)),   y+rr*np.sin(theta) + active_wdw_size*(-np.cos(theta)+np.sin(theta))])
    
    # For each obstacle...
    for i in range(len(ox)):
        obx = ox[i]
        oby = oy[i]
        # ... check if the point is located inside the rectangle defined by c1, c2, c3, c4. Test: M is inside the rectangle ABCD iff 0 < AM.AB < AB.AB and 0 < AM.AD < AD.AD 
        v_ref_hor = np.array([c1[0]-c2[0], c1[1]-c2[1]])
        v_ref_vert = np.array([c3[0]-c2[0], c3[1]-c2[1]])
        vtest = np.array([obx-c2[0], oby-c2[1]])
        if 0<np.dot(vtest, v_ref_hor) and np.dot(vtest, v_ref_hor)<np.dot(v_ref_hor, v_ref_hor) and 0<np.dot(vtest, v_ref_vert) and np.dot(vtest, v_ref_vert)<np.dot(v_ref_vert, v_ref_vert):
            # if the obstacle is visible
            d = np.sqrt((x-obx)**2 + (y-oby)**2)
            fix = f_cr * rr**n * (x-obx) / d**(n+1)
            fiy = f_cr * rr**n * (y-oby) / d**(n+1)
            fx += fix
            fy += fiy
            pmap[int(obx)][int(oby)] = np.sqrt(fix**2 + fiy**2)

    #return the resulting force
    return pmap, np.array([fx,fy]), c1, c2, c3, c4

# Computes the angular command passed to the robot
def command_angular(theta,f):
    # Potential force direction
    delta = math.atan2(f[1], f[0])
    cmd = k*(delta-theta)
    if np.abs(cmd)>omega_max:
        return omega_max * np.sign(cmd)
    return cmd

# Defines the next motion direction
def get_motion_model(theta, f):
    omega = command_angular(theta,f)
    motion = np.array([np.cos(theta + omega), np.sin(theta + omega)])
    return omega, motion

# The actual simulation function
def potential_field_planning(sx, sy, stheta, gx, gy, ox, oy, reso, rr):
    # Initialization
    d = np.sqrt((sx - gx)**2+ (sy - gy)**2)
    rx, ry, rtheta = [sx], [sy], [stheta]
    romega = []

    # While we are too far away from the target...
    while d >= reso:
        # Find the current configuration
        x = rx[-1]
        y = ry[-1]
        theta = rtheta[-1]

        # Compute the potential and the command
        pmap, f, c1, c2, c3, c4 = calc_potential_field(x, y, theta, gx, gy, ox, oy, reso, rr)
        omega, motion = get_motion_model(theta, f)
        romega.append(omega)

        # Update the robot's pose
        v = min(v_max, max(v_max/2,d-v_max))
        xp = x + motion[0]*v
        yp = y + motion[1]*v
        theta = theta + omega
        rx.append(xp)
        ry.append(yp)
        rtheta.append(theta)
        d = np.sqrt((gx - xp)**2+ (gy - yp)**2)

        # Plot part
        if show_animation:
            #plt.plot([c1[0], c2[0], c3[0], c4[0], c1[0]],[c1[1], c2[1], c3[1], c4[1], c1[1]])
            plt.plot(gx, gy, "*m")
            for i in range(len(ox)):
                plt.plot(ox[i], oy[i], "*k")
            plt.plot(xp, yp, ".r")
            plt.pause(0.01)

    print("Goal!!")

    return rx, ry, romega

# Launch function
def main():
    # Initialization
    print("potential_field_planning start")

    sx = 0.0  # start x position [m]
    sy = 15.0  # start y positon [m]
    theta = 0.0
    gx = 30.0  # goal x position [m]
    gy = 30.0  # goal y position [m]
    grid_size = 0.5  # potential grid size [m]
    robot_radius = 0.8 # robot radius [m]
    
    # Obstacle positions
    ox = [15.0, 22.0, 20.0, 25.0, 20.0,15.0, 7.0, 7.0, 25.0]  
    oy = [22.0, 22.0, 26.0, 25.0, 23.0, 23.0, 21.0, 17.0, 27.0] 

    if show_animation:
        plt.grid(True)
        plt.axis("equal")

    # path generation
    rx, ry, romega = potential_field_planning(
        sx, sy, theta, gx, gy, ox, oy, grid_size, robot_radius)

    if show_animation:
        plt.show()


if __name__ == '__main__':
    print(__file__ + " start!!")
    main()
    print(__file__ + " Done!!")