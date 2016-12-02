#!/usr/bin/python3
# -*- coding: utf-8 -*-
import getopt, math, sys, os
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib import ticker
import numpy as np
import sympy as syp
from scipy.stats import gaussian_kde
from tqdm import tqdm

def usage():
    '''Usage function'''
    print(""""Plot 2 dimensional phasespace

Usage: %s -h -i [ifile]

-h                  Show this help message and exit
-i [ifile]          GPT file with SCREEN information x y z rxy Bx By Bz G t ... and position groups.
                    Default is "beam" resulting in "./beam.dat"
""" %sys.argv[0])


def readpositions(file):
    ''' Read in lines as list of strings'''
    print("Reading from %s ... \n" % file)
    f = open(file, 'r')
    lines = []
    #          0      1         2         3         4             5             6             7             8
    #lines = [["id", r"x / m", r"y / m", r"z / m", r"$\beta_x$", r"$\beta_y$", r"$\beta_z$", r"$B_x$ / T", r"$B_y$ / T", r"$B_z$ / T", r"$E_x$ / V/m", r"$E_y$ / V/m", r"$E_z$ / V/m", r"t / s"]]
    positions = False
    caption = True
    for line in f:
        if not line.split(): # Catch empty lines
            continue
        if line.split()[0] == "position":
            pos = float(line.split()[1])
            positions = True
            continue
        if positions != True:
            continue
        else:
            try:
                floats = [float(f) for f in line.split()]
                lines.append([pos] + floats)
            except ValueError:
                if line.split()[0] == "acceptance":
                    break
                if caption != True:
                    lines = [["s / m"] + line.split()] + lines
                    caption = True
    f.close()
    return lines


def recalc(coords):
    '''Calculate phasespace coordinates'''
    print(r"Calculating x, x', y, y', z, z' ... \n")
    #                 1           2           3           4
    lines = [["pos", r"$x$ / m", r"$y$ / m", r"$z$ / m", r"$r_{xy}$ / m", r"$x'$ / rad", r"$y'$ / rad", r"$z'$ / rad", r"$\gamma$", r"$t$ / s", "id"]]
    for f in tqdm(coords[1:]):
        xprime = f[5] / math.sqrt(f[5]**2 + f[6]**2 + f[7]**2)
        yprime = f[6] / math.sqrt(f[5]**2 + f[6]**2 + f[7]**2)
        zprime = f[7] / math.sqrt(f[5]**2 + f[6]**2 + f[7]**2)
        line = f[0:5] + [xprime, yprime, zprime] + f[8:10] + [f[14]]
        lines.append(line)
    return lines


def sort4d(data, column):
    '''Sort multidimensional data according to one column'''
    print("Sorting data by column %i ... \n" % column)
    dict = {}
    for key in tqdm(set(l[column] for l in data[1:])): #set generates unsorted list of all unique values in list
        dict[key] = [data[0]]
        for line in data[1:]:
            if key == line[column]:
                dict[key].append(line)
    return dict



def readlines(file):#, start):
    '''Read in lines as list of string, matched to read CS parameters from GPT output'''
    f = open(file, 'r')
    xlines = []
    ylines = []
    caption = False
    for line in f:
        if line.strip():
            try:
                xlines.append([float(num) for num in [line.split()[i] for i in [0, 15, 17, 18, 19]]])
                ylines.append([float(num) for num in [line.split()[i] for i in [0, 16, 20, 21, 22]]])
            except ValueError as e:
                #print(str(e))
                if caption != True:
                    xlines = [line.split()[15:]] + xlines
                    ylines = [line.split()[15:]] + ylines
                    caption = True
    f.close
    return xlines, ylines


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


def plotellipse(emit, alpha, beta, gamma):
    xmax = math.sqrt(beta*emit)
    ymax = math.sqrt(gamma*emit)
    xrange = np.linspace(-xmax, xmax, num=101)
    yrange = np.linspace(-ymax, ymax, num=101)
    x, y = np.meshgrid(xrange, yrange)
    el = gamma*x**2 + 2*alpha*x*y + beta*y**2
    plt.contour(x, y, el, [emit], colors='r', linewidths=1)
    return


def plot2d(pos, data, column1, xname, column2, yname):
    print("Plotting %s vs. %s at z = %.3f m" % (xname, yname, pos))
    x = [-0.015, 0.015]#[-1e-7, 1e-7]
    y = [-2e-3, 2e-3]#[-1e-7, 1e-7]
    for point in data:
        x.append(float(point[column1]))
        y.append(float(point[column2]))
    cmap = cm.viridis
    hist = ax.hist2d(x, y, bins=(100, 100), cmin=0.001, cmap=cmap, label = "%s-%s-Phasespaxe at %s m" % (xname, yname, pos))#cmin=0.001,
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    cb = f.colorbar(hist[3], ax = ax)
    cb.locator = ticker.MaxNLocator(nbins=6)
    cb.update_ticks()
    return


def main(argv):
    '''read in CMD arguments'''
    fname = "refpart"
    ellipse = ""
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
            
    '''read positions and phasespace coordinates'''
    ifile = "." + os.sep + fname + ".dat"
    coordinates = readpositions(ifile)
    phasespace = recalc(coordinates)
    '''sort by screen position'''
    sort = sort4d(phasespace, 0)
    
    '''read Courant-Snyder Parameters'''
    jfile = "." + os.sep + fname + "_screen-averages.dat"
    if os.path.exists(jfile):
        csx, csy = readlines(jfile)
        csdicts = [sort4d(csx, 0), sort4d(csy, 0)]
        ellipse = 1
         
    '''Plot setup'''
    odir = "." + os.sep + fname + os.sep
    if not os.path.exists(odir): 
        os.makedirs(odir)
    matplotlib_init()

    '''Select which columns to plot'''
    for i, j in [[1, 5], [2, 6]]:#[1, 2]
        for key in sorted(sort.keys()):
            global f, ax
            if ellipse:
                cs = csdicts[i-1]
            f = plt.figure()
            ax = f.add_subplot(111)
            '''Plot data'''
            plot2d(key, sort[key][1:], i, sort[key][0][i], j, sort[key][0][j])
            if ellipse:
                plotellipse(cs[key][1][1], cs[key][1][2], cs[key][1][3], cs[key][1][4])
            ax.locator_params('x', nbins = 6)

            patch = [patches.Patch(color=cm.viridis(0.5))]
            label = [r"%s-%s-Phasespace at s = %s m" % (sort[key][0][i].split(" / ")[0], sort[key][0][j].split(" / ")[0], key)]
            if ellipse:        
                patch.append(lines.Line2D([], [], c = 'r', ls = '-'))
                label.append(r"$\epsilon_{RMS}$")
            ax.legend(patch, label)
                
            plt.tight_layout()
            plt.draw()
            #plt.show()

            '''Save plot'''
            name = sort[key][0][i].split(" / ")[0] + "-" + sort[key][0][j].split(" / ")[0]
            for c in [" ", "$", "\mathcal", "}", "{", "|", "\\", "_", "^"]:
                name = name.replace(c, "")
            name = name.replace("'", "prime") + "_phasespace_%s" %key
            plt.savefig(odir + fname + "_" + name + ".pdf")
            plt.savefig(odir + fname + "_" + name + ".png")
            plt.close('all')
        
if __name__ == "__main__":
    main(sys.argv[1:])

