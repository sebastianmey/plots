#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
Created on Sun Oct 25 13:19:22 2015
@author:    Sebastian Mey, Institut für Kernphysik, Forschungszentrum Jülich
            s.mey@fz-juelich.de
'''

import getopt, math, sys, os
from cycler import cycler
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as cl
import numpy as np
import pickle
from tqdm import tqdm


def usage():
    '''
    Usage function
    '''
    print("""Plot particle path in phacespace.

Usage: %s -h -i [basename] -l -s

-h                  Show this help message and exit
-i [basename]       Basename of ASCII file with trajectories in columns:
                    x y z Bx By Bz Ex Ey Ez betax betay betaz.
                    Default is "refpart", looking up "./refpart_trajectory.dat"
-l                  Load pickeled Python dictionary from instead of sorting ASCII file.
                    Filename is given with the "-i" option.
                    Default is "refpart", looking up "./refpart_trajectory.p
-s                  Save sorted dictionary as pickle to "./basename_trajectory.p"
""" %sys.argv[0])


def readlines(file):
    '''
    Read in lines as list of strings
    '''
    print("Reading from %s ..." % file)
    f = open(file, 'r')
    lines = []
    caption = True
    for line in f:
        if not line.split(): # Catch empty lines
            continue
        if line.split()[0] == "ID": # Capture particle IDs
            id = int(float(line.split()[1]))
            continue
        try:
            floats = [float(f) for f in line.split()]
            lines.append([int(id)] + floats)
        except ValueError:
            if caption != True:
                lines = [["id"] + line.split()] + lines
                caption = True
            continue
    f.close()
    return lines


def recalc(coords):
    '''
    Calculate phasespace trajectory
    '''
    print(r"Calculating x, x', y, y', z, z' ...")
    c=299792458
    #         0                  1            2          3
    lines = [["id", r"$s$ / m", r"$x$ / m", r"$y$ / m", r"$z$ / m", r"$x'$ / rad", r"$y'$ / rad", r"$z'$ / rad$"]]
    for f in tqdm(coords):
        s = f[13] * c * math.sqrt(f[4]**2 + f[5]**2 + f[6]**2)
        xprime = f[4] / math.sqrt(f[4]**2 + f[5]**2 + f[6]**2)
        yprime = f[5] / math.sqrt(f[4]**2 + f[5]**2 + f[6]**2)
        zprime = f[6] / math.sqrt(f[4]**2 + f[5]**2 + f[6]**2)
        line = [f[0], s] + f[1:4] + [xprime, yprime, zprime]
        lines.append(line)
    return lines


def sort4d(data, column):
    '''
    Sort multidimensional data according to one column
    '''
    print("Sorting data by column %i ..." % column)
    dict = {}
    for key in tqdm(set(l[column] for l in data[1:])): #set generates unsorted list of all unique values in list
        dict[key] = [data[0]]
        for line in data[1:]:
            if key == line[column]:
                dict[key].append(line)
    return dict


def matplotlib_init():
    '''Matplotlib settings, so changes to local ~/.config/matplotlib aren't necesarry.'''
    plt.rcParams['mathtext.fontset'] = 'stixsans' # Sans-Serif
    plt.rcParams['mathtext.default'] = 'regular' # No italics
    plt.rcParams['axes.formatter.use_mathtext'] = 'True' # MathText on coordinate axes
    plt.rcParams['axes.formatter.limits'] = (-3, 4) # Digits befor scietific notation
    plt.rcParams['font.size'] = 12.
    plt.rcParams['legend.fontsize'] = 10.
    plt.rcParams['legend.framealpha'] = 0.5
    #plt.rcParams['figure.figsize'] = 8.27, 11.69# A4, for full-page plots
    #plt.rcParams['figure.figsize'] = 8.27, 11.69/2# A5 landscape, for full-width plots in text
    plt.rcParams['figure.figsize'] = 11.69/2, 8.27/2 # A6 landscape, for plots in 2 columns
    #plt.rcParams['figure.figsize'] = 8.27/2, 11.69/2/2# A7 landscape, for plots in 3 columns
    plt.rcParams['figure.dpi'] = 300 # Display resolution
    plt.rcParams['savefig.dpi'] = 300 # Savefig resolution
    plt.rcParams['savefig.format'] = 'pdf'
    plt.rcParams['savefig.transparent'] = True
    plt.rcParams['errorbar.capsize'] = 2
    return


def grayify(cmap):
    '''
    Return a grayscale version of the colormap
    '''
    #cmap = cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    '''convert RGBA to perceived grayscale luminance cf. http://alienryderflex.com/hsp.html'''
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]
    return cl.LinearSegmentedColormap.from_list(cmap.name + "_grayscale", colors, cmap.N)


def setcolor(axis, cmap, nlines):
    colors = []
    for i in np.linspace(0., 0.75, nlines/1):
            colors.append(cmap(i))
    axis.set_prop_cycle(cycler('color', colors))
    #axis.set_color_cycle(colors)
    

def plot3d(id, data, column1, xname, column2, yname, column3, zname):
    x = []
    y = []
    z = []
    for point in data:
        x.append(float(point[column1]))
        y.append(float(point[column2]))
        z.append(float(point[column3]))
    ax.plot(y, x, z, linewidth=.5, zorder=id, alpha=0.9)#100001
    ax.set_ylabel(xname, labelpad=10)
    ax.set_xlabel(yname)
    ax.set_zlabel(zname)
    return [min(x), min(y), min(z)], [max(x), max(y), max(z)], [[x, z], [y, z], [y, x]]
   

def main(argv):
    '''read in CMD arguments'''
    fname = "refpart"
    loaddict = ""
    savedict = ""
    try:                                
        opts, args = getopt.getopt(argv, "hi:ls")
    except getopt.GetoptError as err:
        print(str(err) + "\n")
        usage()                      
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            usage()
            sys.exit()
        elif opt == "-i":
            fname = arg
        elif opt == "-l":
            loaddict = 1
        elif opt == "-s":
            savedict = 1

    '''Get data''' 
    ifile = "." + os.sep + fname + "_trajectory"
    if loaddict:
        print("Reading from %s_trajectory.p ..." % ifile)
        sort = pickle.load(open(ifile + "_traj.p", "rb" ))
    else:
        coordinates = readlines(ifile + ".dat")
        phasespace = recalc(coordinates)
        '''sort 4d-file by particle id into dictionary'''
        sort = sort4d(phasespace, 0)
    if savedict:
        print("Saving Dictionary to %s_trajectory.p ..." % ifile)
        pickle.dump(sort, open(ifile + "_traj.p", "wb" ))
        
    '''Plot setup'''
    dir = "." + os.sep + "test" + os.sep + fname + os.sep
    if not os.path.exists(dir): 
        os.makedirs(dir)
    matplotlib_init()

    '''Select which columns to plot'''
    for i, j in [[2, 3], [2, 5], [3, 6]]:           
        global f, ax
        proj = []
        f = plt.figure()
        ax = f.add_subplot(111, projection='3d', rasterized=True)
        #f.suptitle("%s along beam axis" % field3d[0][j].split("/")[0])
        cmap = (cm.viridis)
        setcolor(ax, cmap, len(sort.keys()))

        '''Plot data'''
        print("Plotting %s vs %s ..." % (sort[1][0][i], sort[1][0][j]))
        mins = []
        maxs = []
        proj = []
        for key in tqdm(sorted(sort.keys(), reverse=True)):
            '''skip everything except refparticle'''
            #if key != 1:
            #    continue
            minimum, maximum, projection = plot3d(key, sort[key][1:], i, sort[key][0][i], 4, sort[key][0][1], j, sort[key][0][j])
            #mins.append(minimum)
            #maxs.append(maximum)
            proj.append(projection)
        ax.autoscale(False)
        xmin, xmax = ax.get_xlim3d()
        ymin, ymax = ax.get_ylim3d()
        zmin, zmax = ax.get_zlim3d()
        #ax.set_xlim(xmin, xmax)
        cmap = grayify(cm.viridis)#CMRmap)
        setcolor(ax, cmap, 3*len(sort.keys()))
        for p in proj:
            ax.plot([xmin] * len(p[0][0]), p[0][0], p[0][1])
            ax.plot(p[1][0], [ymax] * len(p[0][0]), p[1][1])
            ax.plot(p[2][0], p[2][1], [zmin] * len(p[0][0]))

        plt.tight_layout()
        plt.draw()
        #plt.show()

        '''Save plot'''
        name = sort[key][0][i].split(" / ")[0] + "-" + sort[key][0][j].split(" / ")[0]
        for c in [" ", "$", "\mathcal", "}", "{", "|", "\\", "_", "^"]:
            name = name.replace(c, "")
        name = name.replace("'", "prime") + "_traj"
        
        plt.savefig(dir + fname + "_" + name + ".pdf")
        plt.savefig(dir + fname + "_" + name + ".png")
        plt.close('all')

if __name__ == "__main__":
    main(sys.argv[1:])
