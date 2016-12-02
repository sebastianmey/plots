#!/usr/bin/python3
# -*- coding: utf-8 -*-
import getopt, math, sys, os#, subprocess, ROOT
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
    print("""Plot field distribution

    Usage: %s -h -B [Bfile] -E [Efile] - n [float] -o [oname] -s

-h           Show this help message and exit
-B [Bfile]   File with B-Field in columns x y z Bx By Bz
-E [Efile]   File with E-Field in columns x y z Ex Ey Ez
-n [float]   Normalizing factor for fields (e.g. current)
-o [oname]   Optinal absolute path and filename for output, default is "fields", saving to "./fields.dat" 
-s           Save fields to ASCII file
""" %sys.argv[0])


def readlines(file):
    '''
    Read in lines as list of strings
    '''
    print("Reading from %s ..." % file)
    f = open(file, 'r')
    lines = []
    for line in f:
        if line.strip(): # Catch empty lines
            try:
                lines.append([float(num) for num in line.split()])
            except ValueError as e: # Catch lines not containinf floats (e.g. Captions)
                #print(str(e))
                continue
    f.close
    return lines


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
    plt.rcParams['figure.figsize'] = 8.27, 11.69/2# A5 landscape, for full-width plots in text
    #plt.rcParams['figure.figsize'] = 11.69/2, 8.27/2 # A6 landscape, for plots in 2 columns
    #plt.rcParams['figure.figsize'] = 8.27/2, 11.69/2/2# A7 landscape, for plots in 3 columns
    plt.rcParams['figure.dpi'] = 300 # Display resolution
    plt.rcParams['savefig.dpi'] = 300 # Savefig resolution
    plt.rcParams['savefig.format'] = 'pdf'
    plt.rcParams['savefig.transparent'] = True   
    plt.rcParams['errorbar.capsize'] = 2
    return


def join(Bfield, Efield, Norm = 1., Ecorr = 1.):
    #           0            1          2           3             4             5             6               7
    lines = [[r"$x$ / m", r"$y$ / m", r"$z$ / m", r"$B_x$ / T", r"$B_y$ / T", r"$B_z$ / T", r"$E_x$ / V/m", r"$E_y$ / V/m", r"$E_z$ / V/m"]]#, r"$Z$ / $\Omega$"]]
    for b, e in tqdm(zip(Bfield, Efield)):
        l = [B for B in b[0:3]] + [B / Norm for B in b[3:6]] + [Ecorr / Norm * E for E in e[3:6]]#+ [-4e-7*math.pi*float(e[4])/(float(b[3]))]
        lines.append(l)
    return lines


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


def norm(minimum, maximum):
    vmin = min([minimum, -1. * maximum])
    vmax = max([-1. * minimum, maximum])
    return cl.Normalize(vmin, vmax)
    

def plot3d(cut, data, column1, xname, column2, yname, column3, zname, zmult = 1.):
    print("Plotting %s at %.3f m..." % (zname, cut))
    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')#, rasterized = True)
    #f.suptitle("%s at y = %.2f m" % (zname.split(" / ")[0], cut))
    x = []
    y = []
    z = []
    for point in tqdm(data):
        if point[1] != cut:
            continue
        x.append(point[column1])
        y.append(point[column2])
        z.append(zmult * point[column3])
    '''Reshape date for contour plots on coorsinate planes'''
    u = [[], [], []]
    n = [0, 0, 0]
    mn = [0, 0, 0]
    mx = [0 ,0, 0]
    for i, d in enumerate([x, y, z]):
        u[i] = set(d)
        n[i] = len(u[i])
        mn[i] = min(u[i])
        mx[i] = max(u[i])
    xs = np.array(x).reshape(n[0], n[1])
    ys = np.array(y).reshape(n[0], n[1])
    zs = np.array(z).reshape(n[0], n[1])
    '''Common scale fot plots of components of same physicla quantity'''
    global zmin, zmax
    if zname[1] == "B":
        zmin = -6.e-5
        zmax = 4.e-5 
    elif zname[1] == "E":
        zmin = -2.e3
        zmax = 8.e3
    else:
        zmin = mn[2]
        zmax = 500.#mx[2]
    '''Plot data'''
    cmap = cm.viridis
    ax.set_zlim3d(zmin, zmax)
    #scat = ax.scatter(y, x, z, marker = '+', c = zs, cmap=cm.CMRmap, norm = norm(mn[2], mx[2]))
    surf = ax.plot_surface(ys, xs, zs, rstride = 1, cstride = 1, alpha = 0.75, cmap = cmap, norm = norm(zmin, zmax), linewidth=0.01)
    ax.contour(ys, xs, zs, zdir='y', offset = mx[0], levels = np.linspace(mn[0], mx[0], 100), cmap=grayify(cmap), linewidths = .5)
    ax.contour(ys, xs, zs, zdir='x', offset = mn[1], levels = np.linspace(mn[1], mx[1], 100), cmap=grayify(cmap), linewidths = .5)
    ax.contour(ys, xs, zs, zdir='z', offset = zmin, levels = np.linspace(mn[2], mx[2], 100), cmap=cmap, norm = norm(zmin, zmax), linewidths = .5)
    ax.set_ylabel(xname)
    ax.set_xlabel(yname, )
    ax.set_zlabel(zname)
    '''Colorbar'''
    #cb = plt.colorbar(scat, shrink = 0.5)
    cb = plt.colorbar(surf, shrink = 0.5)
    cb.update_ticks()
    #cb.ax.set_ylabel(zname)
    return f, ax 


def main(argv):
    '''Rread in CMD arguments'''
    Bfile = ""
    Efile = ""
    Fnorm = 1.#25.
    save = ""
    ofile = "." + os.sep + "fields"
    try:                                
        opts, args = getopt.getopt(argv, "hB:E:l:n:o:s" )
    except getopt.GetoptError as err:
        print(str(err) + "\n")
        usage()                      
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            usage()
            sys.exit()
        elif opt == "-B":
            Bfile = arg        
        elif opt == "-E":
            Efile = arg
        elif opt == "-n":
            Fnorm = float(arg)
        elif opt == "-o":
            ofile = arg
        elif opt == "-s":
            save = 1

    '''Get data'''
    if Bfile:
        B = readlines(Bfile)
        field3d = B
    if Efile:
        E = readlines(Efile)
        field3d = E
    if Bfile and Efile:
        field3d = join(B, E, Fnorm, 1.0544)
    ys = sorted(set([l[1] for l in field3d[1:]]))
    if save:
        print("Saving ASCII data to %s.dat ..." % ofile)
        f = open(ofile +".dat", 'w')
        f.write("\t".join([string.replace("$", "").replace(" ", "") for string in field3d[0]]) + "\n")
        for line in field3d[1:]:
            f.write("\t".join(map(str, line)) + "\n")
        f.close()
                
    '''Plot data'''
    dir = "." + os.sep + ofile + os.sep
    if not os.path.exists(dir): 
        os.makedirs(dir)
    matplotlib_init()
    print(ys)
    for i, cut in enumerate(ys):
        '''skip everything except y=0 plane'''
        if cut != 0.:#in [-0.03, 0.03]:
            continue
        print(cut)
        for j in range(3, len(field3d[0])):
            '''Get column name'''
            zname = field3d[0][j]
            try:
                name = zname.split(" / ")[0]
                unit = zname.split(" / ")[1]
            except IndexError:
                name = zname
                unit = ""
            plot3d(cut, field3d[1:], 0, field3d[0][0], 2, field3d[0][2], j, zname)
        
            plt.tight_layout()
            plt.draw()
            #plt.show()

            '''Save plot'''
            if cut < 0.:
                fcut = "-" + ("%.3f" % cut)[4:]
            else:
                fcut = ("%.3f" %cut)[3:]
            fname = name.replace("$", "").replace("_", "")
            fname = ofile + "_%s_y%s_%s" % (fname.replace("'", "prime"), fcut, i)
            
            plt.savefig(dir + fname + ".pdf")
            plt.savefig(dir + fname + ".png")
            plt.close('all')

if __name__ == "__main__":
    main(sys.argv[1:])
