#!/usr/bin/python3
# -*- coding: utf-8 -*-
import getopt, math, sys, os
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib import cm
import numpy as np
from tqdm import tqdm

def usage():
    """Usage function"""
    print(""""Plot coordinates and field integrals along each particles trajectory

Usage: %s -h -i [ifile]

-h                  Show this help message and exit
-i [ifile]          File with trajectories in columns x y z Bx By Bz Ex Ey Ez betax betay betaz, default "./trajectory.dat"
""" %sys.argv[0])


def readlines(file):
    '''Read in lines as list of strings'''
    print("Reading from %s ..." % file)
    f = open(file, 'r')
    lines = []
    caption = False
    for line in f:
        if not line.split():
            continue
        if line.split()[0] == "ID":
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


def wien(field):
    '''Calculate Lorentz Force from E and B field'''
    print("Calculating LORENTZ-condition...")
    q=1
    c=299792458
    #         0                1         2         3                                                          7             8             9             10              11
    wien = [["id", r"s / m", r"x / m", r"y / m", r"z / m", r"$x^'$ / rad", r"$y^'$ / rad", r"$z^'$ / rad", r"$B_x$ / T", r"$B_y$ / T", r"$B_z$ / T", r"$E_x$ / V/m", r"$E_y$ / V/m", r"$E_z$ / V/m", r"$F_{Bx}$ / eV/m", r"$F_{By}$ / eV/m", r"$F_{Bz}$ / eV/m", r"$F_{Ex}$ / eV/m", r"$F_{Ey}$ / eV/m", r"$F_{Ez}$ / eV/m", r"$F_x$ / eV/m", r"$F_y$ / eV/m", r"$F_z$ / eV/m"]]
    for f in tqdm(field[1:]):
        s = f[13] * c * 0.459#math.sqrt(f[4]**2 + f[5]**2 + f[6]**2)
        xprime = f[4] / math.sqrt(f[4]**2 + f[5]**2 + f[6]**2)
        yprime = f[5] / math.sqrt(f[4]**2 + f[5]**2 + f[6]**2)
        zprime = f[6] / math.sqrt(f[4]**2 + f[5]**2 + f[6]**2)
        FBx = q*c * (f[5]*f[9] - f[6]*f[8])
        FBy = q*c * (f[6]*f[7] - f[4]*f[9])
        FBz = q*c * (f[4]*f[8] - f[5]*f[7])
        FEx = q * f[10]
        FEy = q * f[11]
        FEz = q * f[12]
        Fx = FEx + FBx
        Fy = FEy + FBy
        Fz = FEz + FBz
        HBx = math.sqrt(f[8]**2 + f[9]**2)/math.sqrt(f[7]**2 + f[8]**2 + f[9]**2)
        HEy = math.sqrt(f[10]**2 + f[12]**2)/math.sqrt(f[10]**2 + f[11]**2 + f[12]**2)
        line = [f[0], s] + f[1:4] + [xprime, yprime, zprime] + f[7:13] + [FBx, FBy, FBz, FEx, FEy, FEz, Fx, Fy, Fz]
        wien.append(line)
    return wien


def sort4d(data, column):
    '''sort data ccording to one column'''
    print("Sorting data by column %i ..." % column)
    dict = {}
    '''set generates a unsorted list of all unique values in column of data'''
    for key in tqdm(set(l[column] for l in data)):
        for line in data:
            if key == line[column]:
                '''if dictionary entry exists, append line, else create new entry'''
                try:
                    dict[int(key)].append(line)
                except KeyError:
                    dict[int(key)] = [line]
    return dict


def setcolor(axis, cmap, ncolumn):
    colors = []
    for i in np.linspace(0., 0.75, ncolumn):
        colors.append(cmap(i))
    axis.set_color_cycle(colors)


def sumtrace(data, column):
    x = data[1][2]
    y = data[1][3]
    z = data[1][1]
    lsum = 0.
    fsum = 0.
    llist = []
    fieldlist = []
    sumlist = []
    for line in data[1:]:
        dx = line[2] - x
        x = line[2]
        dy = line[3] - y
        y = line[3]
        dz = line[1] - z
        z = line[1]
        dl = math.sqrt(dx**2+dy**2+dz**2)
        fdl = line[column] * dl
        lsum = lsum +dl
        fsum = fsum + fdl
        if lsum <= 2.:
            fieldlist.append(line[column])
            llist.append(lsum)
            sumlist.append(fsum)
        else:
            print("sum failed for particle:", line[0])
    return llist, fieldlist, sumlist
        

def plot2d(particle, data, column, name):
    x, y, inty = sumtrace(data, column)
    #ax.plot(x, y, linewidth = .5, ls = '--')
    ax.plot(x, inty, linewidth = .5)
    ax.set_xlabel(r"$l = \sum_i \sqrt{x_i^2+y_i^2+z_i^2}$ / m")
    ax.set_ylabel(r"$\int$%sdl / %s$\cdot$m" % (name.split(" / ")[0],  name.split(" / ")[1]))
    return x[0], x[-1], inty[-1]


def mean(list):
    '''calculate arithmetic average and standard error of a list and prepare scientific notated string output'''
    mean = np.mean(list)
    std = np.std(list)
    if math.fabs(mean) > 1e4 or math.fabs(mean) < 1e-3:
        sci = "%.3e" % mean
        strmean = r"$%s \times 10^{%.0f}$" % (sci.split("e")[0], float(sci.split("e")[1]))
    else:
        strmean = "%.3f" % mean
    if (math.fabs(std) > 1e4 or math.fabs(std) < 1e-3) and std != 0.:
        sci = "%.e" % std
        strstd = r"$%s \times 10^{%.0f}$" % (sci.split("e")[0], float(sci.split("e")[1]))
    else:
        strstd = "%.3f" %std
    return mean, std, strmean, strstd


def main(argv):
    '''read in CMD arguments'''
    fname = "refpart"    
    ifile = "." + os.sep + "trajectory.dat"
    try:                                
        opts, args = getopt.getopt(argv, "hi:")
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
            
    ifile = ".." + os.sep + fname + "_trajectory.dat"
    trajectories = readlines(ifile)
    field3d = wien(trajectories)   
    '''sort 4d-file by particle id into dictionary'''
    field2d = sort4d(field3d[1:], 0)
        
    '''Plot data'''
    dir = "." + os.sep + fname + os.sep
    if not os.path.exists(dir): 
        os.makedirs(dir)
    plt.rcParams['font.size'] = 12
    plt.rcParams['savefig.format'] = 'pdf'
    plt.rcParams['mathtext.default'] = 'regular'
    for j in range(8, len(field3d[0])):
        '''select what to plot'''
        global f, ax
        f = plt.figure()
        #f.suptitle("%s integrated along beam trajectory" % field3d[0][j].split("/")[0])
        ax = f.add_subplot(111)
        setcolor(ax, cm.CMRmap, len(field2d.keys()))
        ints = []
        print("Plotting int %s ..." % field3d[0][j])
        for i, key in enumerate(sorted(field2d.keys())):
            lmin, lmax, int = plot2d(key, field2d[key], j, field3d[0][j])
            ints.append(int)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(-3, 3))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
        ax.xaxis.major.formatter._useMathText = True
        ax.yaxis.major.formatter._useMathText = True
        avg, sigma, stravg, strsigma = mean(ints)
        #Fdl = lines.Line2D([], [], c = 'k', ls = '--')
        intFdl = lines.Line2D([], [], c = cm.CMRmap(0.5), ls = '-')
        #ax.legend([Fdl, intFdl], [r"%s dl / %s$\cdot$m" % (field3d[0][j].split(" /")[0], field3d[0][j].split(" /")[1]), r"$\langle\int_{%.1f m}^{%.01f m}$ %s dl\rangle$ = %s $%s$\cdot$m" % (lmin, lmax, field3d[0][j].split(" /")[0], stravg, field3d[0][j].split(" /")[1])], fancybox=True, framealpha=0.5) # manual legend
        ax.legend([intFdl], [r"$\int_{%.0f m}^{%.0f m}$%sdl = (%s $\pm$ %s) %s$\cdot$ m" % (lmin, lmax, field3d[0][j].split(" /")[0], stravg, strsigma, field3d[0][j].split(" /")[1])], fancybox=True, framealpha=0.5, fontsize='10') # manual legend
        #plt.tight_layout()
        #plt.show()
        name = field3d[0][j].split(" /")[0]
        for c in ["$", "{", "}", "_{"]:
            name = name.strip(c)
        name = name.replace("^'", "prime")
        name = name.replace("_{", "_") + "_int"
        plt.savefig(dir + fname + "_" + name + ".pdf", dpi = 300, orientation = 'landscape', papertype = 'a4', transparent = True)
        plt.savefig(dir + fname + "_" + name + ".png", dpi = 300, orientation = 'landscape', papertype = 'a4', transparent = True)
        plt.close('all')

if __name__ == "__main__":
    main(sys.argv[1:])
