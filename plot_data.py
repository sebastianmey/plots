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


def enge(x=1., A=0.04, L=500., k=.03, x_0 = 1., **kwargs):
    '''Fit function, overload to get label string instead of the function.'''
    expr = "A * (1/(1+np.exp(-k*(L/2-(x-x_0)))) + 1/(1+np.exp(-k*(L/2+(x-x_0)))) - 1)"
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

    
def scipyfit(model, xdata, ydata, yerrdata):
    '''Perform fit and calculate chisquare.'''
    ymax = max(ydata)
    init = (ymax, 500., 10., 0.03)
    '''See http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.curve_fit.html'''
    fitpar, fitcovar = curve_fit(model, xdata, ydata, p0 = init, sigma = yerrdata)
    '''Variance of fit paramaters on the diagonal elements of the covariance matrix'''
    fitparerrs = np.sqrt(np.diag(result.covar))
    chisq = 0.
    for x, y, yerr in zip(xdata, ydata, yerrdata):
        chisq += ((model(x, fitpar[0], fitpar[1], fitpar[2], fitpar[3]) - y)/yerr)**2#, fitpar[3]
    redchisq = chisq / (len(xdata)-len(fitpar))
    print("RedChiSq: %f" % redchisq)
    '''Command line output'''
    for val, err in zip(fitpar, fitparerr):
        print(r"%f $\pm$ %f" % (val, err))
    result.best_values = fitpar
    result.stderrs = fitparerrs
    result.redchi = redchisq
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
    x = np.array(x)
    xerr = np.array(len(x) * [2.])#xerr
    y = np.array(y)
    yerr = np.array(yerr)#len(y) * [0.03])
    '''x values for function plotting'''
    xvals = np.linspace(x[0], x[-1], 100)
    '''Plot measurement'''    
    measurement = ax.errorbar(x, y, xerr = xerr, yerr = yerr, ls = '', label = "Measurement")#yname.split(" / ")[0])
    '''Fit data'''
    #redchisq, fitpar, fitparerr = fit(model, x, y, yerr)
    #fitstrings = ", ".join([r"$\chi^2_{red}=%.1f$" % redchisq,
    #                        r"$B_0=%.4f\pm%.4f$" % (fitpar[0], fitparerr[0]),
    #                        r"$L=%.0f\pm%.0f$" % (fitpar[1], fitparerr[1]),
    #                        r"$x_0=%.1f\pm%.1f$" % (fitpar[2], fitparerr[2]),
    #                        r"$k=%.4f\pm%.4f$" % (fitpar[3], fitparerr[3])])
    #function, = ax.plot(xvals, [model(val, fitpar[0], fitpar[1], fitpar[2], fitpar[3]) for val in xvals], c=measurement[0].get_color(), label = sp.latex(sp.sympify(model(l=True)), mode='inline', mul_symbol='dot'))
    fit = lmfit(model, x, y, yerr)
    text = ", ".join([r"red. $\chi^2$ = %.1f" % fit.redchi] + ["%s = %.4g $\pm$ %.1g" % (key, fit.best_values[key], fit.stderrs[key]) for key in sorted(fit.best_values.keys())])
    fittext = plt.figtext(0., 1., text, fontsize = 8, color = measurement[0].get_color(), horizontalalignment = 'left')
    '''Plot fitted model'''
    function = ax.plot(xvals, fit.best_function(xvals), c=measurement[0].get_color(), label = sp.latex(sp.sympify(model(l=True)), mode='inline', mul_symbol='dot', fold_frac_powers=True, inv_trig_style='full'))
    '''Plot 1sigma intervall'''
    if fit.sigmaband:
        sigma = ax.fill_between(xvals, fit.best_function(xvals) + fit.sigmaband(xvals), fit.best_function(xvals) - fit.sigmaband(xvals), edgecolor = '', facecolor = measurement[0].get_color(), alpha = 0.1, label = r"$1\sigma$ Intervall")
    ax.set_xlim(xvals[0], xvals[-1])
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
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
            
    data = readlines(fname + ".csv")

    matplotlib_init()
    global f, ax
    f = plt.figure()
    ax = f.add_subplot(111)
    
    fittext = []
    fittext.append(fitdata(enge, data, [1, r"z / mm", 10.], [12, r"$B_x$ / mT", 1.]))
    if len(fittext)*0.04-0.02 > 0.06:
        fittext_height = len(fittext)*0.04-0.02
    else:
        fittext_height = 0.06
    for i, text in enumerate(fittext):
        text.set_position((0.04, 1.-(i+1)*0.04))

    h, l = ax.get_legend_handles_labels()
    if len(l) == len(fittext)*3:
        '''Debug'''
        print(l)
        ax.legend((h[2], (h[0], h[1])), (l[2], l[0]), loc=0) # 1 fit
        #ax.legend((h[4], (h[0], h[2]), h[5], (h[1], h[3])), (l[4], l[0], l[5], l[1]), loc=0)# 2 fits
    else:
        ax.legend(loc=0)   
    #ax.set_xlim(0.,130.)
    ax.set_ylim(-0.015,0.05)

    plt.tight_layout(rect=(0,0,1,1-fittext_height))
    plt.draw()
    #plt.show()
    plt.savefig("." + os.sep + odir + oname + ".pdf")
    plt.savefig("." + os.sep + odir + oname + ".png")
    plt.close('all')
                
if __name__ == "__main__":
    main(sys.argv[1:])
