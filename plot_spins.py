#!/usr/bin/python3
# -*- coding: utf-8 -*-
import getopt, math, sys, os
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib import cm
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pickle
from tqdm import tqdm


def usage():
    '''
    Usage function
    '''
    print(""""Plot spin motion along each particles trajectory

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


def tbmt(data):
    '''
    Calculate the spin angular frequency vector from E and B field
    '''
    print("Calculating Omega...")
    q = 1
    c = 299792458
    m = 1875612928 # eV/c²
    G = 0.8574382311 - 1 # NIST recommended value
    t = 0.
    id = 0
    S = np.array([1., 0., 0.])
    #         0      1         2         3           4             5             6            7             8             9             10              11              12
    lines = [["id", r"x / m", r"y / m", r"z / m", r"$\beta_x$", r"$\beta_y$", r"$\beta_z$", r"$B_x$ / T", r"$B_y$ / T", r"$B_z$ / T", r"$E_x$ / V/m", r"$E_y$ / V/m", r"$E_z$ / V/m", r"t / s", r"s / m", r"$\Omega_x$ / Hz", r"$\Omega_y$ / Hz", r"$\Omega_z$ / Hz", r"$S_x$",  r"$S_y$", r"$S_z$"]]
    for f in tqdm(data):
        if f[0] != id:
            t = 0.
            S = np.array([0., 0., 1.]) # reset spin vector for every new particle
        id = f[0]
        beta = np.array([f[4], f[5], f[6]])
        B = np.array([f[7], f[8], f[9]])
        E = np.array([f[10], f[11], f[12]])
        absbeta = np.sqrt(np.dot(beta, beta))
        gamma = 1 / np.sqrt(1 - absbeta**2)
        s = f[13] * c * absbeta
        dt = f[13] - t
        t = f[13]
        Omega = -q / m * c**2 * ((1/gamma+G) * B - (gamma*G/(gamma+1)) * np.dot(beta, B) * beta - (1/(gamma+1)+G)/c * np.cross(beta, E))
        absOmega = np.sqrt(Omega[0]**2 + Omega[1]**2 + Omega[2]**2)
        if absOmega == 0.:
            continue
        axis = 1/absOmega * Omega
        angle = absOmega*dt
        '''Rotation of S around axis about ang with quarternions, equivalent to (0, S') = (cos(angle/2), sin(angle/2)*axis)*(0, S)*(cos(angle/2), -sin(angle/2)*axis)'''
        S = S + 2*np.cos(angle/2)*np.sin(angle/2)*np.cross(axis, S) + 2*np.sin(angle/2)*np.sin(angle/2)*np.cross(axis, np.cross(axis, S))
        #S = S + (1-np.cos(angle))*np.cross(axis, np.cross(axis, S)) + np.sin(angle)*np.cross(axis, S)
        #print("unit quarternion? q^2 =", np.cos(angle/2) - np.sin(angle/2)/absOmega*np.dot(Omega,Omega))
        #print("spin is unit vector? sqrt(S^2) = ", np.sqrt(np.dot(S, S)))
        #print(Omega[0], Omega[1], Omega[2])
        #print(S[0], S[1], S[2])
        f = f + [s, Omega[0], Omega[1], Omega[2], S[0], S[1], S[2]]
        lines.append(f)
    return lines


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
    plt.rcParams['figure.figsize'] = 8.27/2, 11.69/2/2# A7 landscape, for plots in 3 columns
    plt.rcParams['figure.dpi'] = 300 # Display resolution
    plt.rcParams['savefig.dpi'] = 300 # Savefig resolution
    plt.rcParams['savefig.format'] = 'pdf'
    plt.rcParams['savefig.transparent'] = True
    plt.rcParams['errorbar.capsize'] = 2
    return


def sort4d(data, column):
    '''
    Sort multidimensional data according to one column
    '''
    print("Sorting data by column %i ..." % column)
    dict = {}
    for key in tqdm(set(l[column] for l in data[1:])): # set generates unsorted list of all unique values in list
        dict[key] = [data[0]]
        for line in data[1:]:
            if key == line[column]:
                dict[key].append(line)
    return dict


def setcolor(axis, cmap, ncolumn):
    colors = []
    for i in np.linspace(0., 0.75, ncolumn):
        colors.append(cmap(i))
    axis.set_prop_cycle(cycler('color', colors))
    #axis.set_color_cycle(colors)# deprecated
    return
    

def plotcolumn(data, column, name, id):
    x = []
    y = []
    #y2 = []
    for l in data:
        x_l = l[3]#[14]
        y_l = l[column]
        #if y_l == 0.:
        #    print("Recheck particle:", l[0])
        #y2_l = l[column+2]
        x.append(x_l)
        y.append(y_l)
        #y2.append(y2_l)
    ax.plot(x, y, linewidth = .5, alpha = 0.7, zorder = -id)#, ls = ":")
    #ax.plot(x, y2, linewidth = .5, alpha = 0.7)
    ax.set_xlabel(r"s / m")
    ax.set_ylabel(name)# "S"
    ax.autoscale()
    #ymin, ymax = ax.get_ylim()
    #ax.set_ylim(ymin*1.1, ymax-1.1)
    #ax.set_xlim(x[0], x[-1])
    return y[-1]#, y2[-1]


def mean(list):
    '''
    Calculate arithmetic average and standard error of a list and prepare scientific notated string output
    '''
    mean = np.mean(list)
    std = np.std(list)
    if math.fabs(mean) > 1e4 or math.fabs(mean) < 1e-3:
        sci = "%.3e" % mean
        strmean = r"$%s \times 10^{%.0f}$" % (sci.split("e")[0], float(sci.split("e")[1]))
    elif 1.-mean < 1e-3:
        sci = "%.3e" % (1.-mean)
        strmean = r"1.0 - $%s \times 10^{%.0f}$" % (sci.split("e")[0], float(sci.split("e")[1]))
    else:
        strmean = "%.3f" % mean
    if (math.fabs(std) > 1e4 or math.fabs(std) < 1e-3) and std != 0.:
        sci = "%.e" % std
        strstd = r"$%s \times 10^{%.0f}$" % (sci.split("e")[0], float(sci.split("e")[1]))
    else:
        strstd = "%.3f" %std
    return mean, std, strmean, strstd


def main(argv):
    '''Read in CMD arguments'''
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
        print("Reading from %s_spin.p ..." % ifile)
        sort = pickle.load(open(ifile + "_spin.p", "rb" ))
    else:
        trajectories = readlines(ifile + ".dat")
        field3d = tbmt(trajectories)
        '''sort 4d-file by particle id into dictionary'''
        sort = sort4d(field3d, 0)#[1:]
    if savedict:
        print("Saving Dictionary to %s_spin.p ..." % ifile)
        pickle.dump(sort, open(ifile + "_spin.p", "wb"))      

    '''Plot setup'''
    odir = "." + os.sep + "Sz" + os.sep + fname + os.sep
    if not os.path.exists(odir):
        os.makedirs(odir)
    matplotlib_init()
    '''Select which columns to plot'''
    for j in range(18, len(sort[2][0])):#[19]:
        global f, ax, axhist
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.set_rasterization_zorder(0)
        #f.suptitle("%s integrated along beam trajectory" % field3d[0][j].split("/")[0])                       

        '''Get column names'''
        fullname = sort[2][0][j]
        try:
            name = fullname.split(" / ")[0]
            unit = fullname.split(" / ")[1]
        except IndexError:
            name = fullname
            unit = ""
        #fullname2 = sort[2][0][j+2]
        #try:
        #    name2 = fullname2.split(" / ")[0]
        #    unit2 = fullname2.split(" / ")[1]
        #except IndexError:
        #    name2 = fullname2
        #    unit2 = ""

        '''Plot data'''
        cmap = cm.viridis#_r
        setcolor(ax, cmap, len(sort.keys()))
        spins = []
        #spins2 = []
        print("Plotting %s ..." % name)
        for key in tqdm(sort.keys()):
            spin = plotcolumn(sort[key][1:], j, fullname, key)#, spin2
            spins.append(spin)
            #spins2.append(spin2)
        avg, sigma, strmean, strsigma = mean(spins)
        #avg2, sigma2, strmean2, strsigma2 = mean(spins2)

        '''Legend'''
        ls = [lines.Line2D([], [], c = cmap(0.5), ls = '-')]
        #ls = [lines.Line2D([], [], c = cmap(0.5), ls = ':'), lines.Line2D([], [], c = cmap(0.5), ls = '-')]
        if j > 17:
            names = [r"%s = %s %s" % (name, strmean, unit) + "\n" + r"$\sigma$(%s) = %s %s" %(name, strsigma, unit)]
            #names = [r"%s = %s %s" % (name, strmean, unit) + "\n" + r"$\sigma$(%s) = %s %s" %(name, strsigma, unit), r"%s = %s %s" % (name2, strmean2, unit2) + "\n" + r"$\sigma$(%s) = %s %s" %(name2, strsigma2, unit2)]
        else:
            names = [name]
        ax.legend(ls, names, loc = 0) # manual legend
        ax.locator_params('x', nbins = 5)
        ax.locator_params('y', nbins = 5)

        if j == 20:
            def mjrFormatter(x, pos):
                return "%.0f" % ((float(x) - 1.)*1e11)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(mjrFormatter))
            ax.text(0., 1.01, r"$\times 10^{-11} + 1.0$", fontsize=12, transform = ax.transAxes)
        
        '''Rrighthanded distribution histogram'''
        if j > 17:
            divider = make_axes_locatable(ax)
            axhist = divider.append_axes('right', 0.9, pad = 0.1, sharey=ax)
            plt.setp(axhist.get_yticklabels(), visible=False)      
            n, bins, patches = axhist.hist(spins, 100, histtype='bar', orientation ='horizontal')#normed = 1,
            #n2, bins2, patches2 = axhist.hist(spins2, 100, histtype='bar', orientation ='horizontal')#normed = 1,                     
            colors = [cmap(i) for i in 1-n/max(n)]#np.linspace(0., 0.75, n)]
            for (c,p) in zip(colors, patches):
                plt.setp(p, fc=c, ec=c)
            #for (c,p) in zip(colors, patches2):
                #plt.setp(p, fc=c, ec=c)
            axhist.locator_params('x', nbins = 3)
            axhist.set_xlim(0, 400)
            #axhist.set_xlabel(name)
            #axhist.set_xlabel("# Particles")


        plt.tight_layout(rect=(0,0,1,0.93))
        plt.draw()
        #plt.show()

        '''Save plot'''
        for c in ["$", "{", "}", "^", "\\", "_"]:
            name = name.replace(c, "")
            #name2 = name2.replace(c, "")
        name = name.replace("'", "prime")# + "_" + name2.replace("'", "prime") + "_spin"
        plt.savefig(odir + fname + "_" + name + "_spin.pdf")
        plt.savefig(odir + fname + "_" + name + "_spin.png")
        plt.close('all')

if __name__ == "__main__":
    main(sys.argv[1:])
