"""

Potential Field based path planner

author: Atsushi Sakai (@Atsushi_twi)

Ref:
https://www.cs.cmu.edu/~motionplanning/lecture/Chap4-Potential-Field_howie.pdf

"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
n = 2
F_ct = 5.0  # attractive potential gain
F_cr = 100.0  # repulsive potential gain
AREA_WIDTH = 30.0  # potential area width [m]

show_animation = True


def calc_potential_field(gx, gy, ox, oy, reso, rr):
    minx = min(ox) - AREA_WIDTH / 2.0
    miny = min(oy) - AREA_WIDTH / 2.0
    maxx = max(ox) + AREA_WIDTH / 2.0
    maxy = max(oy) + AREA_WIDTH / 2.0
    xw = int(round((maxx - minx) / reso))
    yw = int(round((maxy - miny) / reso))

    # calc each potential
    pmap = [[0.0 for i in range(yw)] for i in range(xw)]

    for ix in range(xw):
        x = ix * reso + minx

        for iy in range(yw):
            y = iy * reso + miny
            ug = calc_attractive_potential(x, y, gx, gy)
            uo = calc_repulsive_potential(x, y, ox, oy, rr)
            uf = ug + uo
            pmap[ix][iy] = uf

    return pmap, minx, miny


def calc_attractive_potential(x, y, gx, gy):
    d = np.hypot(x - gx, y - gy)
    #print(d)
    return F_ct * np.array([(gx-x)/d, (gy-y)/d])


def calc_repulsive_potential(x, y, ox, oy, rr):
    # search nearest obstacle
    F_r = np.array([0,0])
    for i in range(len(ox)):
        d = np.hypot(x - ox[i], y - oy[i])
        f_x = F_cr * rr **n / d ** (n+1) * (ox[i] - x)
        f_y = F_cr * rr **n / d ** (n+1) * (oy[i] - y)
        F_r +=  np.array([f_x, f_y])

    return F_r


def get_motion_model():
    # dx, dy
    motion = [[1, 0],
              [0, 1],
              [-1, 0],
              [0, -1],
              [-1, -1],
              [-1, 1],
              [1, -1],
              [1, 1]]

    return motion


def potential_field_planning(sx, sy, gx, gy, ox, oy, reso, rr):

    # calc potential field
    pmap, minx, miny = calc_potential_field(sx, sy, ox, oy, reso, rr)

    # search path
    d = np.hypot(sx - gx, sy - gy)
    ix = round((sx - minx) / reso)
    iy = round((sy - miny) / reso)
    gix = round((gx - minx) / reso)
    giy = round((gy - miny) / reso)

    #if show_animation:
    if True:
        draw_heatmap(pmap)
        plt.plot(ix, iy, "*k")
        plt.plot(gix, giy, "*m")

    rx, ry = [sx], [sy]
    motion = get_motion_model()
    #while d >= reso:
     #   minp = float("inf")
      #  minix, miniy = -1, -1
       # for i in range(len(motion)):
        #    inx = int(ix + motion[i][0])
         #   iny = int(iy + motion[i][1])
          #  if inx >= len(pmap) or iny >= len(pmap[0]):
           #     p = float("inf")  # outside area
            #else:
             #   p = pmap[inx][iny]
            #if minp > p:
             #   minp = p
              #  minix = inx
               # miniy = iny
       # ix = minix
        #iy = miniy
        #xp = ix * reso + minx
        #yp = iy * reso + miny
        #d = np.hypot(gx - xp, gy - yp)
        #rx.append(xp)
        #ry.append(yp)

        #if show_animation:
         #   plt.plot(ix, iy, ".r")
    plt.pause(2)

    print("Goal!!")

    return rx, ry


def draw_heatmap(data):
    tab = []
    for i in range(len(data)):
        tabi = []
        for j in range(len(data[i])):
            tabi.append(np.hypot(data[i][j][0], data[i][j][1]))
        tab.append(tabi)
    data = np.array(tab).T
    plt.pcolor(data, vmax=100.0, cmap=plt.cm.Blues)


def main():
    print("potential_field_planning start")

    sx = 0.0  # start x position [m]
    sy = 10.0  # start y positon [m]

    gx = 30.0  # goal x position [m]
    gy = 30.0  # goal y position [m]
    grid_size = 0.5  # potential grid size [m]
    robot_radius = 5.0  # robot radius [m]

    ox = [15.0, 5.0, 20.0, 25.0, 17.5, 22.5]  # obstacle x position list [m]
    oy = [25.0, 15.0, 26.0, 25.0, 25.5, 25.5]  # obstacle y position list [m]

    if show_animation:
        plt.grid(True)
        plt.axis("equal")

    # path generation
    rx, ry = potential_field_planning(
        sx, sy, gx, gy, ox, oy, grid_size, robot_radius)

    print(rx)
    print(ry)

    if show_animation:
        plt.show()


if __name__ == '__main__':
    print(__file__ + " start!!")
    main()
    print(__file__ + " Done!!")
