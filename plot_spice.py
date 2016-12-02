#!/usr/bin/python3
# -*- coding: utf-8 -*-
import getopt, math, os, sys, subprocess
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
import numpy as np


def usage():
    """Usage function"""
    print("""Usage: %s  -h -i [ifile] -o [ofile]

-h                     Show this help message and exit
-i [ifile]             Baseame of input file exported from LTSpice
-o [ofile]             Optional output filename, default is the same as ifile
""" %sys.argv[0])

    
def readlines(file):
    # Read in lines as list of strings, format columns into lists
    f = open(file, 'r', encoding = 'latin_1')
    lines = {}
    #cap = [cname for cname in f.readline().split()]
    cap = [r"f / Hz", r"$\hat U$ / V", r"$\hat I$ / A", r"$S_{11}$ / dB", r"$S_{21}$ / dB", r"$Z_{in}$ / $\Omega$"]
    for line in f:
        if line.strip(): # Filter out empty lines
            if line.split()[0] == "Step":
                cs = float(line.split()[2].split("=")[1][:-1])
                lines[cs] = [cap]
            try:
                point = [float(line.split()[0])] #frequency
                for entry in line.split()[1:]:
                    pair = entry[1:-1].split(",")
                    amp = float(pair[0][:-2])
                    phase = float(pair[1][:-1])
                    point.append([amp, phase]) #amp, phase for each measurement
                lines[cs].append(point)
            except ValueError as e:
                #print(str(e))
                continue
    f.close()
    return lines


def dB2field(dB):
    return 10**(dB / 20)


def grayify(cmap):
    """Return a grayscale version of the colormap"""
    #cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    # convert RGBA to perceived greyscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]
    return cmap.from_list(cmap.name + "_grayscale", colors, cmap.N)


def setcolor(cmap, ncolumn):
    colors = []
    for i in np.linspace(0.3, 0.9, ncolumn):
        colors.append(cmap(i))
    return colors


def plot2d(cpdict, columns):
    print("Plotting...")
    cmaps = {}
    for key, cmap in zip(sorted(cpdict.keys()), [cm.Purples, cm.Blues, cm.Greens, cm.Reds]):
        cmaps[key] = cmap
    x = []
    xname = columns[0][1]
    # fill x from first entry
    for line in cpdict[200.][250.][1:]:
        x.append(line[columns[0][0]] * float(columns[0][2]))
    plt.xlabel(xname)
    
    for ax, column in zip(axarr, columns[1:]):
        ax2 = ax.twinx()
        l = {}
        l2 = {}
        cphandles = []
        cshandles = []
        for cp, csdict in sorted(cpdict.items()):
            colors = setcolor(cmaps[cp], len(sorted(csdict.keys())))
            grays = {cs:gray for cs, gray in zip(sorted(csdict.keys()), setcolor(grayify(cmaps[cp]), len(sorted(csdict.keys()))))}
            ax.set_color_cycle(colors)
            l[cp] = {}
            l2[cp] = {}            
            for cs, data in sorted(csdict.items()):
                yamp = []
                yphase = []
                yname = column[1]
                for line in data[1:]:
                    if yname[0:2] == "$S":
                        yamp.append(line[column[0]][0] * float(column[2]))
                    else:
                        yamp.append(dB2field(line[column[0]][0]) * float(column[2]))
                    yphase.append(line[column[0]][1])

                l[cp][cs], = ax.plot(x, yamp, label = r"$C_p$=%.0fpF, $C_s$=%.0fpF" % (cp, cs))
                if yname[0:2]  == "$Z":
                    ax.set_yscale('log')
                    
                c = l[cp][cs].get_color()
                lw = plt.getp(l[cp][cs], 'linewidth')
                l2[cp][cs], = ax2.plot(x, yphase, ls = ':', linewidth = lw / 2, c = c)
                if cs == 550.:
                    cphandles.append(l[cp][cs])
                if cp == 200:
                    cshandles.append(lines.Line2D([], [], c = grays[cs], ls = '-'))

        amp = lines.Line2D([], [], c = 'k', ls = '-')
        phase = lines.Line2D([], [], c = 'k', ls = ':')
        ax2.legend([amp, phase], ["%s" % yname.split(" / ")[0], r"$\phi$(%s)" % yname.split(" / ")[0]], fancybox=True, framealpha=0.5, prop={'size':8}, loc = 1)#handlelength=2.5, for dashed lines

        ax.ticklabel_format(style='sci', axis='x', scilimits=(-3, 3))
        ax.xaxis.major.formatter._useMathText = True

        ax.set_ylabel(r"%s" % yname)
        #ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
        ax.yaxis.major.formatter._useMathText = True

        ax2.set_ylabel(r"$\phi$ (%s)  / °" % yname.split(" / ")[0])
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
        ax2.yaxis.major.formatter._useMathText = True

    cplabels_rfb = ["$C_p$=200pF","$C_p$=700pF","$C_p$=1200pF","$C_p$=1700pF"]
    cplabels_rfe = ["$C_p$=200pF","$C_p$=1700pF","$C_p$=3200pF","$C_p$=4700pF"]
    firstlegend = axarr[0].legend(cphandles[::-1], cplabels_rfe[::-1], fancybox=True, framealpha=0.5, prop={'size':10}, bbox_to_anchor=(0., 1.35, 1., .102), loc=3, ncol=len(cphandles), mode="expand", borderaxespad=0.)# use for rf-e
    axarr[0].add_artist(firstlegend)

    secondlegend = axarr[0].legend(cshandles[::-1], ["$C_s$=250pF","$C_s$=300pF","$C_s$=350pF","$C_s$=400pF","$C_s$=450pF","$C_s$=500pF","$C_s$=550pF"][::-1], fancybox=True, framealpha=0.5, prop={'size':8}, bbox_to_anchor=(0., 1.18, 1., .102), loc=3, ncol=len(cshandles), mode="expand", borderaxespad=0.)
    axarr[0].add_artist(secondlegend)
    
    plt.xlim(.5e6, 1.5e6)
    return cshandles, cphandles


def main(argv):
    # read in CMD arguments
    fname = ""
    try:                                
        opts, args = getopt.getopt(argv, "hci:o:d:g:t:")
    except getopt.GetoptError as err:
        print(str(err) + "\n")
        usage()                      
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            usage()
            sys.exit()
        elif opt == "-i":
            fname = str(arg)
            oname = fname
        elif opt == "-o":
            oname = str(arg)
    cpdict = {}
    #for cp in range(200, 1701, 500):#use for rf-b
    for cp in range(200, 4701, 1500):#use for rf-e
        csdict = readlines(fname + "_Cp%s.txt" % cp)
        cpdict[float(cp)] = csdict
    
    plt.rcParams['font.size'] = 12
    plt.rcParams['savefig.format'] = 'pdf'
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rc('figure', figsize=(8.27,11.69,))
    global f, axarr
    f, axarr = plt.subplots(len(cpdict[1700.][550.][0]) - 1, 1, sharex = 'col')#plot for each column in data except the frequencies 
    #f.suptitle(r"RF B parameters for discrete values of $C_p$ and $C_s$")
    cslines, cplines = plot2d(cpdict, [[i, entry, 1] for i, entry in enumerate(cpdict[200.][250][0])]) # [column number, column name, scaling] for every column in cs
    #plt.tight_layout()
    plt.savefig("." + os.sep + oname + ".pdf", dpi = 300, orientation = 'portrait', papertype = 'a4', transparent = True)
    plt.savefig("." + os.sep + oname + ".png", dpi = 300, orientation = 'portrait', papertype = 'a4', transparent = True)
    plt.show()
    plt.close('all')
    
if __name__ == "__main__":
    main(sys.argv[1:])
