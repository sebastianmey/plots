#!/usr/bin/python3
# -*- coding: utf-8 -*-
import getopt, math, os, sys, subprocess
from lmfit import Model, Parameters
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import sympy as sp
from tqdm import tqdm


def usage():
    '''
    Usage function
    '''
    print("""Usage: %s  -h -c -i [ifile] -o [ofile] -g [\"x_start y_start x_end y_end\"] -t [fieldtype]

-h                     Show this help message and exit
-c                     Calculate mesh and fields with poisson, otherwise use pregenerated file "ifile.T35"
-i [ifile]             Input name of .am file to process by Poisson
-o [ofile]             Optional output filename, default "ifile_x" and "ifile_y"
-g [\"x0 y0,xf yf\"]   Define field interpolation region between x0 and xf, y0 and yf, enables plotting
-t [type]              Optional type, either "B" or "E", by default the last character in in ifile
""" %sys.argv[0])

    
def readlines(file):#, start):
    '''
    Read in lines as list of strings, cut header
    '''
    f = open(file, 'r')
    lines = []
    print("Reading data from %s..." %file)
    for line in f:
        if line.strip():
            try:
                lines.append([float(num) for num in line.split()])
            except ValueError as e:
                #print(str(e))
                continue
    f.close
    return lines

    
def poisson(basename):
    '''
    Automesh, Poisson and fieldline plot on file basename.
    '''
    print("Calculating mesh...")
    mesh = subprocess.Popen(["env", wineprefix, "wine", progpath + "AUTOMESH.EXE", basename + ".am"])
    mesh.wait()
    print("Calculating fields...")
    fields = subprocess.Popen(["env", wineprefix, "wine", progpath + "POISSON.EXE", basename + ".T35"])
    fields.wait()
    subprocess.Popen(["env", wineprefix, "wine", progpath + "WSFPLOT.EXE", basename + ".T35"])
    return

    
def grid(basename, xstart, ystart, xend, yend):
    '''
    Generate .in7 File and run SF7
    '''
    stepmult = 2.
    steps = [int(abs(xstart - xend)*stepmult), int(abs(ystart - yend)*stepmult*2)]
    print("# steps [x, y]: %s" % steps)
    in7 = open(basename + ".in7", 'w')
    #in7.write("""Grid\n%s %s\n%s %s\nEND""" %(start, end, steps[0], steps[1]))
    if steps[0] != 0:
        in7.write("""Line\n%s %s %s %s\n%s\nEND""" %(xstart, ystart, xend, yend, steps[0]))
        direction = [0, r"x / mm", [xstart, xend]]
    else:
        in7.write("""Line\n%s %s %s %s\n%s\nEND""" %(xstart, ystart, xend, yend, steps[1]))
        direction = [1, r"y / mm", [ystart, yend]]
    in7.close()
    print("Interpolating fields...")
    sf7 = subprocess.Popen(["env", wineprefix, "wine", progpath + "SF7.EXE", basename + ".in7", basename + ".T35"])
    sf7.wait()
    os.rename("OUTSF7.TXT",  basename + "_GRID.TXT")
    return direction


def matplotlib_init():
    '''
    Matplotlib settings, so changes to local ~/.config/matplotlib aren't necesarry
    '''
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


def enge(x=1., A=0.04, L=500., k=.03, **kwargs):
    '''Fit function, overload to get label string instead of the function.'''
    expr = "A * (1/(1+np.exp(-k*(L/2-(x)))) + 1/(1+np.exp(-k*(L/2+(x)))) - 1)"
    if kwargs:
        return expr.replace("np.", "")
    else:
        return eval(expr)
    

def tanh(x=1., A=1., L=1., k=1., x_0=1., **kwargs):
    '''Fit function, overload to get label string instead of the function.'''
    expr = "A/2 * (np.tanh(k/2*(L/2-(x+x_0)))+ np.tanh(k/2*(L/2+(x+x_0))))"
    if kwargs:
        return expr.replace("np.", "")
    else:
        return eval(expr)

    
def lmfit(func, xdata, ydata, yerrdata):
    '''Perform fit and calculate chisquare.'''
    model = Model(func, independent_vars=['x'], missing = 'drop')
    '''Builds symbolic function and its gradient'''
    symbs = [sp.var(v) for v in model.independent_vars + model.param_names]#func.__code__.co_varnames[:-2]]
    function = sp.sympify(func(l=True))
    '''Parameter constrains, if needed'''
    #model.make_params()
    #pars.print_param_hints()
    print(model.param_names)
    '''Fit'''
    result = model.fit(ydata, x=xdata, weights=1/yerrdata)
    print(result.fit_report())
    '''Calculate confidence intervals if standard erros are not enough (e.g. for correlated parameters)'''
    #result.conf_interval()
    #print(result.ci_report())
    '''Calculate standard errors of fit parameters'''
    stderrs = np.sqrt(np.diag(result.covar))
    result.stderrs = {key: stderrs[i] for i, key in enumerate(result.best_values.keys())}
    '''Calculate 1-sigma intervall around fit'''
    grad = sp.Matrix([sp.diff(function, s) for s in result.best_values.keys()])
    M = sp.Matrix(result.covar.tolist())
    gTMg = grad.T * M * grad
    '''Debug'''
    #sp.pprint(grad)
    #sp.pprint(function)
    #sp.pprint(gTMg)
    best_values = [(s, result.best_values[s]) for s in result.best_values.keys()]
    result.best_function = sp.lambdify(x, function.subs(best_values), 'numpy')
    result.sigmaband = sp.lambdify(x, sp.sqrt(gTMg[0]).subs(best_values), 'numpy')
    return result


def fitdata(model, data, xcol, ycol):
    '''
    Plot data in coloumn #datacol with errors in #datacol + 1.
    "col" is an list in the form: [#col, name, scaling]
    '''
    x = []
    xerr = []
    xname = xcol[1]
    y = []
    yerr = []
    yname = ycol[1]
    print("Plotting %s against %s ..." %(yname, xname))
    for line in tqdm(data):
        x.append(line[xcol[0]] * xcol[2])
        #xerr.append(line[xcol[0]] + 1] * xcol[2])
        y.append(line[ycol[0]] * ycol[2])
        yerr.append(line[ycol[0] + 1] * ycol[2])
    x = np.array(x + [-200., 200.])
    xerr = np.array(len(x + 2) * [2.])#xerr
    y = np.array(y + [0.002,0.002])
    yerr = np.array(yerr + [0.0001,0.0001])#len(y) * [0.03])
    '''x values for function plotting'''
    xvals = np.linspace(-125., 125., 100)#x[0], x[-1], 100)
    '''Plot measurement'''    
    measurement = ax.errorbar(x, y, xerr = xerr, yerr = yerr, ls = '', label = "Measurement")#yname.split(" / ")[0])
    '''Fit data'''
    fit = lmfit(model, x, y, yerr)
    text = ", ".join([r"red. $\chi^2$ = %.1f" % fit.redchi] + ["%s = %.4g $\pm$ %.1g" % (key, fit.best_values[key], fit.stderrs[key]) for key in fit.best_values.keys()])
    fittext = plt.figtext(0., 1., text, fontsize = 8, color = measurement[0].get_color(), horizontalalignment = 'left')
    '''Plot fitted model'''
    function = ax.plot(xvals, fit.best_function(xvals), c=measurement[0].get_color(), label = sp.latex(sp.sympify(model(l=True)), mode='inline', mul_symbol='dot', fold_frac_powers=True, inv_trig_style='full'))
    '''Plot 1sigma intervall'''
    if fit.sigmaband:
        sigma = ax.fill_between(xvals, fit.best_function(xvals) + fit.sigmaband(xvals)*2, fit.best_function(xvals) - fit.sigmaband(xvals)*2, edgecolor = '', facecolor = measurement[0].get_color(), alpha = 0.1, label = r"$1\sigma$ Intervall")
    #ax.set_xlim(xvals[0], xvals[-1])
    #ax.set_xlabel(xname)
    #ax.set_ylabel(yname)
    return fittext


def plot2d(data, columns):
    print("Plotting...")
    x = []
    xname = columns[0][1]
    for line in data:
        x.append(float(line[columns[0][0]]) * float(columns[0][2]))
    yname = []
    for column in columns[1:]:
        y = []
        yname = column[1]
        for line in data:
            y.append(float(line[column[0]]) * float(column[2]))
        ax.plot(x, y, label = "Poisson Simulation", ls=':', c='b')#yname.split(" / ")[0]
        #ax.plot(x, y, label = yname, ls=column[3], c=column[4])
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    return


def main(argv):
    # read in CMD arguments
    global wineprefix, progpath
    fishpath = os.path.expanduser("~mey") + os.sep + ".PlayOnLinux/wineprefix/Superfish/"
    progpath = fishpath + "drive_c/LANL/"
    wineprefix = "WINEPREFIX=" + fishpath
    fname = ""
    calculate = 0
    plot = 0
    gstart = ""
    gend = ""
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
        elif opt == "-c":
            calculate = 1
        elif opt == "-i":
            fname = str(arg)
            type = fname[-1]
            oname = fname
        elif opt == "-o":
            oname = str(arg)
        elif opt == "-g":
            xstart = float((arg.split(",")[0]).split()[0])
            ystart = float((arg.split(",")[0]).split()[1])
            xend = float((arg.split(",")[1]).split()[0])
            yend = float((arg.split(",")[1]).split()[1])
        elif opt == "-t":
            type = str(arg)

    if calculate:
        poisson(fname)
    '''Interpolate fields along line'''
    if xstart or ystart:
        direction = grid(fname, xstart, ystart, xend, yend)
        field = readlines(fname + "_GRID.TXT")
        #direction_nf = grid(fname[:-2] + "_NOFERRITES_B", xstart, ystart, xend, yend)
        #field_nf = readlines(fname[:-2] + "_NOFERRITES_B" + "_GRID.TXT")
        if type == "B":
            unit = "mT"
            scale = 0.1
        elif type == "E":
            unit = "V/m"
            scale = 100
        # debug
        #print(type, unit, scale)

        matplotlib_init()
        global f, ax
        f = plt.figure()
        ax = f.add_subplot(111)

        measurement = readlines(fname + "_MEASUREMENT.csv")
        fittext = []
        fittext.append(fitdata(enge, measurement, [0, "x / mm", 1], [11, "Measurement", 1.])) # Overlay measured data
        for i, text in enumerate(fittext):
            text.set_position((0.04, 1.-(i+1)*0.04))
        #plot = plot2d(field, [[direction[0], direction[1], 1], [2, r"$%s_x$ / %s" % (type, unit), scale], [3,  r"$%s_y$ / %s" % (type, unit), scale]])
        plot = plot2d(field, [[direction[0], direction[1], 1], [2, r"$%s_x$ / %s" % (type, unit), scale]])
        #plot = plot2d(field, [[direction[0], direction[1], 1], [2, r"$%s_x$ with ferrites" % type, scale, '-', 'b'], [3,  r"$%s_y$ with ferrites" % type, scale, '-', 'g']])
        #plot = plot2d(field_nf, [[direction_nf[0], direction_nf[1], 1], [2, r"$%s_x$ without ferrites" % type, scale, ':', 'b'], [3,  r"$%s_y$ without ferrites" % type, scale, ':', 'g']])

        #plt.locator_params(axis='x',nbins=5)
        ax.set_xlim(direction[2])
        ax.set_ylim(-0.015, 0.05)
        #ax.set_ylabel(r"$%s$ / %s" % (type, unit))
        h, l = ax.get_legend_handles_labels()
        '''Debug'''
        print(l)
        #ax.legend(loc=4)
        ax.legend((h[3], (h[0], h[2]), h[1]), (l[3], l[0], l[1]), loc=4)

        plt.tight_layout(rect=(0,0,1,0.98))
        plt.draw()
        #plt.show()
        plt.savefig("." + os.sep + "lines_new" + os.sep + oname + "_%s_meas.pdf" % direction[1][0])
        plt.savefig("." + os.sep + "lines_new" + os.sep + oname + "_%s_meas.png" % direction[1][0])
        plt.close('all')

if __name__ == "__main__":
    main(sys.argv[1:])
