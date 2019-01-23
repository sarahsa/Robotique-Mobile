
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
f_ct = 0.8  # attractive potential gain
f_cr = 1.2 # repulsive potential gain
n = 2 # chosen dimension
k = 0.5 # proportional constant for steering
active_wdw_size = 4 # acitve window size [m]
AREA_WIDTH = 30.0  # area width [m]

# Grid parameters
minx = - AREA_WIDTH / 2.0
miny = - AREA_WIDTH / 2.0
maxx = + AREA_WIDTH / 2.0
maxy = + AREA_WIDTH / 2.0
grid_size = 0.5  # potential grid size [m]

# Others parameters
omega_max = 120 #[s**(-1)]
v_max = 1.5 #[m.s^(-1)]
v_min = 0.8 #[m.s^(-1)]
mass = 1.0 #[kg]
robot_radius = 1.5 # robot radius [m]

# Compute the potential for the given robot position
def calc_potential_field(x, y, theta, gx, gy, ox, oy, wx, wy, wlx, wly, wlrotate, reso, rr):
    xw = int(round((maxx - minx) / reso))
    yw = int(round((maxy - miny) / reso))

    # calc each potential
    pmap = [[0.0 for i in range(yw)] for i in range(xw)]

    # sum of the two below defined potentials (attractive and repulsive)
    pmap, ft = calc_attractive_potential(pmap, x, y, gx, gy)
    pmap, fr, c1, c2, c3, c4 = calc_repulsive_potential(pmap, x, y,theta, ox, oy, wx, wy, wlx, wly, wlrotate, rr)
    return pmap, fr+ft, c1, c2, c3, c4

# Attractive potential
def calc_attractive_potential(pmap, x, y, gx, gy):
    d = np.sqrt((x-gx)**2 + (y-gy)**2)
    fx = f_ct * (gx-x) / d
    fy = f_ct * (gy-y) / d
    pmap[int(gx)][int(gy)] = np.sqrt(fx**2+ fy**2)
    return pmap, np.array([fx,fy])

# Repulsive potential
def calc_repulsive_potential(pmap, x, y,theta, ox, oy, wx, wy, wlx, wly, wlrotate, rr):
    fx = 0
    fy = 0
    # Calculate the coordinates of active window corners
    ct = np.cos(theta)
    st = np.sin(theta)
    rct = rr*ct
    rst = rr*st
    c1 = np.array([x+rct + active_wdw_size*st,
                   y+rst - active_wdw_size*ct])
    c2 = np.array([x+rct - active_wdw_size*st,
                   y+rst + active_wdw_size*ct])
    c3 = np.array([x+rct + active_wdw_size*(-st+ct),
                   y+rst + active_wdw_size*(ct+st)])
    c4 = np.array([x+rct + active_wdw_size*(st+ct),
                   y+rst + active_wdw_size*(-ct+st)])

    # For each discrete obstacle...
    for i in range(len(ox)):
        obx = ox[i]
        oby = oy[i]
        # cf below
        if is_Point_in_Rect(obx, oby, c1, c2, c3, c4):
            # if the obstacle is visible
            d = np.sqrt((x-obx)**2 + (y-oby)**2)
            fix = f_cr * rr**n * (x-obx) / d**(n+1)
            fiy = f_cr * rr**n * (y-oby) / d**(n+1)
            fx += fix
            fy += fiy
            pmap[int(obx)][int(oby)] = np.sqrt(fix**2 + fiy**2)

    # For each wall...
    for j in range(len(wx)):
        # compute the corners coordinates
        x1 = wx[j]
        y1 = wy[j]
        l = wlx[j]
        h = wly[j]
        c = np.cos(wlrotate[j])
        s = np.sin(wlrotate[j])
        d1 = np.array([x1,y1])
        d2 = np.array([x1+l*c, y1+l*s])
        d3 = np.array([x1+l*c-h*s, y1+l*s+h*c])
        d4 = np.array([x1-h*s, y1+h*c])
        # define a region of interest around the rotated rectangle
        minx = min(d1[0],d2[0], d3[0], d4[0])
        maxx = max(d1[0],d2[0], d3[0], d4[0])
        miny = min(d1[1],d2[1], d3[1], d4[1])
        maxy = max(d1[1],d2[1], d3[1], d4[1])
        # test if each point of this region is simultaneously located inside the active window and inside the wall. If this is the case, then update the potential force.
        for k1 in range(int(minx), int(maxx)+1):
            for k2 in range(int(miny), int(maxy)+1):
                if is_Point_in_Rect(k1, k2, c1, c2, c3, c4) and is_Point_in_Rect(k1, k2, d1, d2, d3, d4):
                    d = np.sqrt((k1-x)**2+(k2-y)**2)
                    fix = f_cr * rr**n * (x-k1) / d**(n+1)
                    fiy = f_cr * rr**n * (y-k2) / d**(n+1)
                    fx += fix
                    fy += fiy
                    pmap[int(k1)][int(k2)] = np.sqrt(fix**2 + fiy**2)

    #return the resulting force
    return pmap, np.array([fx,fy]), c1, c2, c3, c4

# Test if a point P is inside a given rectangle. Test: P is inside the rectangle ABCD iff 0 < AP.AB < AB.AB and 0 < AP.AD < AD.AD
def is_Point_in_Rect(x, y, c1, c2, c3, c4):
    v_ref_hor = np.array([c1[0]-c2[0], c1[1]-c2[1]])
    v_ref_vert = np.array([c3[0]-c2[0], c3[1]-c2[1]])
    vtest = np.array([x-c2[0], y-c2[1]])
    return 0<np.dot(vtest, v_ref_hor) and np.dot(vtest, v_ref_hor)<np.dot(v_ref_hor, v_ref_hor) and 0<np.dot(vtest, v_ref_vert) and np.dot(vtest, v_ref_vert)<np.dot(v_ref_vert, v_ref_vert)

# Computes the angular command passed to the robot
def command_angular(theta,f):
    # Potential force direction
    delta = math.atan2(f[1], f[0])
    cmd = k*(delta-theta)
    if np.abs(cmd)>omega_max:
        return omega_max * np.sign(cmd)
    return cmd

# Computes the velocity command passed to the robot
def command_velocity(theta,f):
    omega = command_angular(theta, f)
    v1 = min(v_max, min(np.sqrt(f[0]**2+f[1]**2), v_min))/mass
    return v1

# Defines the next motion direction
def get_motion_model(theta, f):
    omega = command_angular(theta,f)
    motion = np.array([np.cos(theta + omega), np.sin(theta + omega)])
    return omega, motion

# The actual simulation function
def potential_field_planning(sx, sy, stheta, gx, gy, reso, rr, ox=[], oy=[], wx=[], wy=[], wlx=[], wly=[], wlrotate=[]):
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
        pmap, f, c1, c2, c3, c4 = calc_potential_field(x, y, theta, gx, gy, ox, oy, wx, wy, wlx, wly, wlrotate, reso, rr)
        omega, motion = get_motion_model(theta, f)
        romega.append(omega)

        # Update the robot's pose
        v = command_velocity(theta,f)
        xp = x + motion[0]*v
        yp = y + motion[1]*v
        theta = theta + omega
        rx.append(xp)
        ry.append(yp)
        rtheta.append(theta)
        d = np.sqrt((gx - xp)**2+ (gy - yp)**2)

        # Plot part
        if show_animation:
            plt.plot(gx, gy, "*m")
            #plt.plot([xp, xp+f[0]],[yp, yp+f[1]])
            for i in range(len(ox)):
                plt.plot(ox[i], oy[i], "*k")
            for k in range(len(wx)):
                x1 = wx[k]
                y1 = wy[k]
                l = wlx[k]
                h = wly[k]
                c = np.cos(wlrotate[k])
                s = np.sin(wlrotate[k])
                plt.plot([x1, x1+l*c, x1+l*c-h*s, x1-h*s, x1],
                         [y1, y1+l*s, y1+l*s+h*c, y1+h*c, y1], "-k")
            plt.plot(xp, yp, ".r")
            plt.pause(0.01)

    print("Goal!!")

    return rx, ry, romega, rtheta

# Launch function
def main():
    # Initialization
    print("potential_field_planning start")

    sx = 5.0  # start x position [m]
    sy = 20.0  # start y positon [m]
    theta = np.pi
    gx = 30.0  # goal x position [m]
    gy = 30.0  # goal y position [m]

    # Discrete bstacles positions
    #ox = [15.0, 22.0, 20.0, 25.0,15.0, 7.0, 7.0, 25.0]
    #oy = [22.0, 22.0, 26.0, 25.0, 23.0, 21.0, 17.0, 27.0]
    ox1 =[15]*70
    ox2 = np.linspace(5, 15, 70)
    ox = np.concatenate((ox1, ox2, ox2))

    oy1 = np.linspace(19, 23, 70)
    oy2 = [23]*70
    oy3 = [19]*70
    oy = np.concatenate((oy1, oy2, oy3))
    print("oy=" , oy)

    # Walls positions and lengths
    wx = [5, 5] #wx: x-coordinate of the bottom-left corner of the wall
    wy = [17.0, 23.0] #wx: y-coordinate of the bottom-left corner of the wall
    wlx = [15, 15]  #wlx: wall length
    wly = [2, 2]   #wly: wall height
    wlrotate = [0, 0]

    if show_animation:
        plt.grid(True)
        plt.axis("equal")

    # path generation
    rx, ry, romega, rtheta = potential_field_planning(
        sx, sy, theta, gx, gy,grid_size, robot_radius, ox, oy, wx, wy, wlx, wly, wlrotate)
    #rx, ry, romega, rtheta = potential_field_planning(
    #    sx, sy, theta, gx, gy,grid_size, robot_radius, ox, oy)

    if show_animation:
        plt.show()


if __name__ == '__main__':
    print(__file__ + " start!!")
    main()
    print(__file__ + " Done!!")
