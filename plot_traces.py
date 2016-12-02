#!/usr/bin/python3
# -*- coding: utf-8 -*-
import getopt, math, sys, os
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pickle
from tqdm import tqdm


def usage():
    '''Usage function'''
    print("""Plot coordinates and field amplitudes along each particles trajectory

Usage: %s -h -i [basename] -l -s

-h                  Show this help message and exit
-i [basename]       Basename of ASCII file with trajectories in columns:
                    x y z Bx By Bz Ex Ey Ez betax betay betaz
                    Default is "refpart", looking up "./refpart_trajectory.dat"
-l                  Load pickeled Python dictionary from instead of sorting ASCII file.
                    Filename is given with the "-i" option.
                    Default is "refpart", looking up "./refpart_trajectory.p
-s                  Save sorted dictionary as pickle to "./basename_trajectory.p"
""" %sys.argv[0])


def readlines(file):
    '''Read in lines as list of strings'''
    print("Reading from %s ..." % file)
    f = open(file, 'r')
    lines = []
    caption = True#False
    for line in f:
        if not line.split(): # Catch empty lines
            continue
        if line.split()[0] == "ID": # Capture particle IDs
            id = int(float(line.split()[1]))
            continue
        if id in [10000]: # last particle sometimes simulated wrong
            continue
        try:
            floats = [float(f) for f in line.split()]
            lines.append([id] + floats)
        except ValueError: # Catch lines not containing floats (e.g. Captions)
            if caption != True:
                lines = [["id"] + line.split()] + lines
                caption = True
            continue
    f.close()
    return lines


def lorentz(field):
    '''Calculate Lorentz Force from E and B field'''
    print("Calculating LORENTZ-condition...")
    q=1
    c=299792458
    #         0                1         2         3                                                      7             8             9             10              11
    lines = [["id", r"s / m", r"x / m", r"y / m", r"z / m", r"$x'$ / rad", r"$y'$ / rad", r"$z'$ / rad", r"$B_x$ / T", r"$B_y$ / T", r"$B_z$ / T", r"$E_x$ / V/m", r"$E_y$ / V/m", r"$E_z$ / V/m", r"$F_{Bx}$ / eV/m", r"$F_{By}$ / eV/m", r"$F_{Bz}$ / eV/m", r"$F_{Ex}$ / eV/m", r"$F_{Ey}$ / eV/m", r"$F_{Ez}$ / eV/m", r"$F_x$ / eV/m", r"$F_y$ / eV/m", r"$F_z$ / eV/m", r"|B| / T", r"|E| / V/m", r"$\sqrt{B_y^2+B_z^2}$ / T", r"$\sqrt{E_x^2+E_z^2}$ / V/m"]]
    for f in tqdm(field):
        s = f[13] * c * math.sqrt(f[4]**2 + f[5]**2 + f[6]**2)
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
        absB = math.sqrt(f[7]**2 + f[8]**2 + f[9]**2)
        absE = math.sqrt(f[10]**2 + f[11]**2 + f[12]**2)
        unwantedB =  math.sqrt(f[8]**2 + f[9]**2)
        unwantedE =  math.sqrt(f[10]**2 + f[12]**2)
        line = [f[0], s] + f[1:4] + [xprime, yprime, zprime] + f[7:13] + [FBx, FBy, FBz, FEx, FEy, FEz, Fx, Fy, Fz, absB, absE, unwantedB, unwantedE]
        lines.append(line)
    return lines


def sort4d(data, column):
    '''Sort multidimensional data according to one column'''
    print("Sorting data by column %i ..." % column)
    dict = {}
    for key in tqdm(set(l[column] for l in data[1:])): # set generates unsorted list of all unique values in list
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
    plt.rcParams['legend.fontsize'] = 8.
    plt.rcParams['legend.framealpha'] = 0.5
    #plt.rcParams['figure.figsize'] = 8.27, 11.69# A4, for full-page plots
    #plt.rcParams['figure.figsize'] = 8.27, 11.69/2# A5 landscape, for full-width plots in text
    #plt.rcParams['figure.figsize'] = 11.69/2, 8.27/2 # A6 landscape, for plots in 2 columns
    plt.rcParams['figure.figsize'] = 8.27/2, 11.69/4# A7 landscape, for plots in 3 columns
    plt.rcParams['figure.dpi'] = 300 # Display resolution
    plt.rcParams['savefig.dpi'] = 300 # Savefig resolution
    plt.rcParams['savefig.format'] = 'pdf'
    plt.rcParams['savefig.transparent'] = True
    plt.rcParams['errorbar.capsize'] = 2
    return


def setcolor(axis, cmap, ncolumn):
    colors = []
    for i in np.linspace(0., 0.75, ncolumn):
        colors.append(cmap(i))
    axis.set_prop_cycle(cycler('color', colors))
    #axis.set_color_cycle(colors) deprecated
    return


def sumtrace(data, column):
    x = data[0][2]
    y = data[0][3]
    z = data[0][4]
    s = data[0][1]
    lsum = 0.
    fsum = 0.
    llist = []
    zlist = []
    fieldlist = []
    sumlist = []
    for line in data[1:-1]:
        dx = line[2] - x
        x = line[2]
        dy = line[3] - y
        y = line[3]
        dz = line[4] - z
        z = line[4]
        ds = line[1] - s
        s = line[1]
        dl = ds#math.sqrt(dx**2+dy**2+dz**2)
        fdl = line[column] * ds#dl
        lsum = lsum + dl
        fsum = fsum + fdl
        #if line[column] == 0.:
        #    print("Recheck particle:", line[0])
        fieldlist.append(line[column])
        zlist.append(z)
        llist.append(lsum)
        sumlist.append(fsum)
    return zlist, llist, fieldlist, sumlist
        

def plotcolumn(data, column, name, id):
    x, s, y, inty = sumtrace(data, column)
    ax.plot(x, y, linewidth=.5, alpha = 0.7, zorder = -id)
    ax.set_xlabel(r"s / m")#= \sum_i c \left|\vec\beta_i\right| t_i$ / m") #\sqrt{x_i^2+y_i^2+z_i^2}
    ax.set_ylabel(name.split(", ")[0])
    ax.autoscale()                      
#    xmin, xmax = ax.get_xlim()
    ax.set_xlim(-1., 1.)#x[0], x[-1])
    return y[-1], inty[-1]


def mean(list):
    '''Calculate arithmetic average and standard deviation of a list and prepare scientific notated string output'''
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
    fame = "refpart"
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
        print("Reading from %s_trace.p ..." % ifile)
        sort = pickle.load(open(ifile + "_trace.p", "rb" ))
    else:
        traces = readlines(ifile + ".dat")
        fields = lorentz(traces)
        '''sort 4d-file by particle id into dictionary'''
        sort = sort4d(fields, 0)
    if savedict:
        print("Saving Dictionary to %s_trace.p ..." % ifile)
        pickle.dump(sort, open(ifile + "_trace.p", "wb" ))
        
    '''Plot setup'''
    odir = "." + os.sep + "test" + os.sep + fname + os.sep #"matplotlib"
    if not os.path.exists(odir): 
        os.makedirs(odir)
    matplotlib_init()
    
    '''Select which columns to plot'''
    for j in [2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26]:
        global f, ax
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.set_rasterization_zorder(0)
        cmap = cm.viridis#_r
        setcolor(ax, cmap, len(sort.keys()))

        '''Get column names'''
        fullname = sort[2][0][j]
        try:
            name = fullname.split(" / ")[0]
            unit = fullname.split(" / ")[1]
        except IndexError:
            name = fullname
            unit = ""

        '''Plot data'''
        youts = []
        yints = []
        print("Plotting %s ..." % name)
        for key in tqdm(sort.keys()):
            yout, yint = plotcolumn(sort[key][1:], j, fullname, key)
            youts.append(yout)
            yints.append(yint)
        outavg, outsigma, stroutavg, stroutsigma = mean(youts)
        intavg, intsigma, strintavg, strintsigma = mean(yints)
        print("%s = %s %s; " % (name, outavg, unit) + "sigma(%s) = %s %s" %(name, outsigma, unit))
        print("%s dl = %s %s; " % (name, intavg, unit) + "sigma(%s dl) = %s %s" %(name, intsigma, unit))

        '''Legend'''
        ls = [lines.Line2D([], [], c = cmap(0.5), ls = '-')]
        if j > 7:
            names = [r"$\int$%sdl = %s %s$\cdot$ m" % (name, strintavg, unit) + "\n" + r"$\sigma(\int$%sdl) = %s %s $\cdot$ m" %(name, strintsigma, unit)]
        else:
            names = [r"%s$_{out}$ = %s %s" % (name, stroutavg, unit) + "\n" + r"$\sigma$(%s$_{out}$) = %s %s" %(name, stroutsigma, unit)]
        ax.legend(ls, names, loc = 0)
                        
        plt.tight_layout()
        plt.draw()
        #plt.show()
        
        '''Save plot'''                
        for c in ["$", "\mathcal", "}", "{", "|", "^", "\\", "_"]:
            name = name.replace(c, "")
        name = name.replace("^'", "prime") + "_trace"
        plt.savefig(odir + fname + "_" + name + ".pdf")
        plt.savefig(odir + fname + "_" + name + ".png")
        plt.close('all')

if __name__ == "__main__":
    main(sys.argv[1:])
