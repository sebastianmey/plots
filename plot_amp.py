#!/usr/bin/python3
# -*- coding: utf-8 -*-
import getopt, math, os, sys, subprocess
from lmfit import Model, Parameters
import matplotlib.pyplot as plt
#import matplotlib.lines as lines
#from matplotlib import cm
from scipy.optimize import curve_fit
import numpy as np
import sympy as sp
from tqdm import tqdm


def usage():
    '''Usage function'''
    print("""Usage: %s  -h -i [ifile] -o [ofile] 

-h                     Show this help message and exit
-i [ifile]             Input name of data file
-o [ofile]             Optional output filename, default "ifile".pdf
""" %sys.argv[0])


def readlines(file):
    '''Read in lines as list of strings.'''
    f = open(file, 'r')
    lines = []
    for line in f:
        '''Catch empty lines'''
        if line.strip():
            try:
                '''Only lines with numbers'''
                lines.append([float(num) for num in line.split()])
            except ValueError as e:
                '''Debug'''
                #print(str(e))
                continue
    f.close
    return lines


def matplotlib_init():
    '''Matplotlib settings, so changes to local ~/.config/matplotlib aren't necesarry.'''
    plt.rcParams['mathtext.fontset'] = 'stixsans' # Sans-Serif
    plt.rcParams['mathtext.default'] = 'regular' # No italics
    plt.rcParams['axes.formatter.use_mathtext'] = 'True' # MathText on coordinate axes
    plt.rcParams['axes.formatter.limits'] = (-3, 3) # Digits befor scietific notation
    plt.rcParams['font.size'] = 12.
    plt.rcParams['legend.fontsize'] = 8.
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


def pol1(x=1, y_0=1., c=1., **kwargs):
    '''Fit function, overload with keyword "l" to get string instead of the callable function.'''
    expr = "y_0 + c * x"
    if 'l' in kwargs.keys():
        return expr.replace("np.", "")
    else:
        return eval(expr)


def fsqrt(x=1, c=1., **kwargs):
    '''Fit function, overload with keyword "l" to get string instead of the callable function.'''
    expr = "np.sqrt(c * x)"
    if 'l' in kwargs.keys():
        return expr.replace("np.", "")
    else:
        return eval(expr)

    
def lmfit(func, xdata, ydata, yerrdata):
    model = Model(func, independent_vars=['x'], missing = 'drop')
    '''Builds symbolic function and its gradient'''
    function = sp.sympify(func(l=True))
    symbs = [sp.var(v) for v in model.independent_vars + model.param_names]#func.__code__.co_varnames[:-2]]
    '''Parameter constrains, if needed'''
    #model.set_param_hint('k', value = 50000., min = 0., vary = True)
    #model.make_params()
    #model.print_param_hints()
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
    #sp.pprint(grad)
    #sp.pprint(function)
    #sp.pprint(gTMg)
    best_values = [(s, result.best_values[str(s)]) for s in result.best_values.keys()]
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
        #xerr.append(line[xcol[0] + 1] * xcol[2])
        y.append(line[ycol[0]] * ycol[2])
        #yerr.append(line[ycol[0] + 1] * ycol[2])
    x = np.array(x)
    #xerr = np.array(len(x) * [0.005])# const
    xerr = x * 0.01# %
    y = np.array(y)
    #yerr = len(y) * [0.]# const
    yerr = y * 0.01# % 
    #yerr = np.sqrt(np.array(yerr)**2 + (y * 0.005)**2)# sqrt((yerr)² + (y * 0.005)²)
    '''x values for function plotting'''
    if axlim:
        xvals = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
    else:
        xvals = np.linspace(x[0], x[-1], 100)
    '''Plot measurement'''    
    measurement = ax.errorbar(x, y, xerr = xerr, yerr = yerr, ls = '', label = yname.split(" / ")[0])
    '''Fit data'''
    fit = lmfit(model, x, y, yerr)
    text = ", ".join([r"$red. \chi^2 = %.2g$" % fit.redchi] + ["$%s = %.3g \pm %.1g$" % (key, fit.best_values[key], fit.stderrs[key]) for key in sorted(fit.best_values.keys())])
    fittext = plt.figtext(0., 1., text, fontsize = 8, color = measurement[0].get_color(), horizontalalignment = 'left')
    '''Plot fitted model'''
    function, = ax.plot(xvals, fit.best_function(xvals), c=measurement[0].get_color(), label = sp.latex(sp.sympify(model(l=True)), mode='inline', mul_symbol='dot'))
    '''Plot 1sigma intervall'''
    if fit.sigmaband:
        sigma = ax.fill_between(xvals, fit.best_function(xvals) + fit.sigmaband(xvals), fit.best_function(xvals) - fit.sigmaband(xvals),edgecolor = '', facecolor = measurement[0].get_color(), alpha = 0.1, label = r"$1\sigma$ Intervall")
    ax.set_xlabel(xname)
    ax.set_xlim(xvals[0], xvals[-1])
    return fittext


def main(argv):
    '''Read in CMD arguments'''
    fname = ""
    odir = "." + os.sep + "plots" + os.sep
    if not os.path.exists(odir): 
        os.makedirs(odir)
    try:                                
        opts, args = getopt.getopt(argv, "hi:o:")
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


    matplotlib_init()
    global f, ax, axlim
    f = plt.figure()
    ax = f.add_subplot(111)
    #f.suptitle(r"Cable Loss RF-E Lower Electrode")

    fittext = []
    if fname == "rf-b_amp":
        f_630 = readlines(fname + ".csv")[:8]
        f_871 = readlines(fname + ".csv")[8:]
        fittext.append(fitdata(fsqrt, f_630, [1, r"$P_{in}$ / W", 1.],[3, r"$\hat I_{RF-B}$ at f = 630 kHz", 0.5]))
        fittext.append(fitdata(fsqrt, f_871, [1, r"$P_{in}$ / W", 1.], [3, r"$\hat I_{RF-B}$ at f = 871 kHz", 0.5]))
        #fittext.append(fitdata(fsqrt, f_630, [1, r"$P_{in,RF-B}$ / W", 1.], [10, r"$\hat U_{RF-E}$ at f = 630 kHz", 1]))
        #fittext.append(fitdata(fsqrt, f_871, [1, r"$P_{in,RF-B}$ / W", 1.], [10, r"$\hat U_{RF-E}$ at f = 871 kHz", 1]))
        ax.set_ylabel(r"I / A")
        #ax.set_ylabel(r"$U_{RF-E} / V$")
        
    elif fname == "rf-e_amp":
        f_630 = readlines(fname + ".csv")[:9]
        f_871 = readlines(fname + ".csv")[9:]
        #fittext.append(fitdata(fsqrt, f_630, [1, r"$P_{in}$ / W", 1.], [9, r"$\hat U_{RF-E}$ at f = 630 kHz", 1.]))
        #fittext.append(fitdata(fsqrt, f_871, [1, r"$P_{in}$ / W", 1.], [9, r"$\hat U_{RF-E}$ at f = 871 kHz", 1.]))
        fittext.append(fitdata(fsqrt, f_630, [1, r"$P_{in,RF-E}$ / W", 1.], [11, r"$\hat I_{RF-B}$ at f = 630 kHz", 1]))
        fittext.append(fitdata(fsqrt, f_871, [1, r"$P_{in,RF-E}$ / W", 1.], [11, r"$\hat I_{RF-B}$ at f = 871 kHz", 1]))
        ax.set_ylabel(r"U / V")
        #ax.set_ylabel(r"$I_{RF-B} / A$")
        
    elif fname == "rf-wien_ring-warte":
        axlim = ax.set_xlim(0.,2600)
        ax.set_ylim(0.,2600)
        cal = readlines(fname + ".csv")
        #fittext.append(fitdata(pol1, cal, [0, r"$I_{RF-B}$ / A (1 M$\Omega$ termination)", 0.5], [2, r"$\hat I_{Control Room}$", 0.5]))
        #fittext.append(fitdata(pol1, cal, [0, r"$I_{RF-B}$ / A (1 M$\Omega$ termination)", 0.5], [1, r"$\hat I_{RF-B}$", 0.5]))
        #ax.set_ylabel(r"I / A (50 $\Omega$ termination)")
        fittext.append(fitdata(pol1, cal, [3, r"$U_{RF-E}$ / V (1 M$\Omega$ termination) / V", 1000.], [5, r"$\hat U_{Control Room}$", 1000.]))
        #fittext.append(fitdata(pol1, cal, [3, r"$U_{RF-E}$ / V (1 M$\Omega$ termination) / V", 1.], [4, r"$\hat U_{Control Room}$", 1.]))
        ax.set_ylabel(r"U / V (50 $\Omega$ termination)")

    if len(fittext)*0.04-0.02 > 0.06:
        fittext_height = len(fittext)*0.04-0.02
    else:
        fittext_height = 0.06
    for i, text in enumerate(fittext):
        text.set_position((0.025, 1.-(i+1)*0.04))

    h, l = ax.get_legend_handles_labels()
    if len(l) == len(fittext)*3:
        '''Debug'''
        print(l)
        #ax.legend((h[4], (h[0], h[2]), h[5], (h[1], h[3])), (l[4], l[0], l[5], l[1]), loc=4)
        ax.legend((h[2], (h[0], h[1])), (l[2], l[0]), loc=4)
    else:
        ax.legend()   

  
    plt.tight_layout(rect=(0,0,1,1-fittext_height))
    plt.draw()
    #plt.show()
    plt.savefig("." + os.sep + odir + oname + ".pdf")
    plt.savefig("." + os.sep + odir + oname + ".png")
    plt.close('all')
                
if __name__ == "__main__":
    main(sys.argv[1:])
