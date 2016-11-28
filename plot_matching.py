#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
Created on Mo Jan 04 08:06:22 2015
@author:    Sebastian Mey, Institut für Kernphysik, Forschungszentrum Jülich
            s.mey@fz-juelich.de
'''

import getopt, math, os, sys, subprocess
from lmfit import Model, Parameters
import matplotlib.pyplot as plt
#import matplotlib.lines as lines
#from matplotlib import cm
import numpy as np
import sympy as sp
from tqdm import tqdm


class symbolic:
    '''
    Take string expression defined in a function and generate symbolic representation in SymPy.
    Calculates Gradient. Value are determined via Lambdify.
    '''
    def __init__(self, function):
        self.string = function(l = True) # String
        self.sympyfunc = sp.sympify(self.string) # SymPy expression
        self.tex = "$" + sp.latex(self.sympyfunc) + "$" # String
        self.variables = function.__code__.co_varnames[:-2] # List of strings
        self.symbols = [sp.var(v) for v in self.variables] #List of SymPy symbols
        self.value = sp.lambdify(self.symbols, self.sympyfunc, 'numpy') # Lambda object -> np.float64
        self.ofx = sp.sympify("0") # SymPy expression
        self.gradfunc = sp.Matrix([sp.diff(self.sympyfunc, s) for s in self.symbols[1:]]) # SymPy expression
        self.gradvalue = sp.lambdify(self.symbols, self.gradfunc, 'numpy') # Lamda object -> np.ndarray
        self.gradofx = sp.sympify("0") #SymPy expression

    def evaluate(self, **kwargs):
        '''Gives a np.float64 value, requires values fot ALL variables'''
        values = [kwargs[v] for v in self.variables]
        return self.value(*values)

    def get_ofx(self, **kwargs):
        '''Gives a function of the inependent variale by substituting parameters for values.'''
        repl = [(s, kwargs[str(s)]) for s in self.symbols if str(s) in kwargs.keys()]
        self.ofx = self.sympyfunction.subs(repl)
        return self.ofx

    def evaluate_grad(self, **kwargs):
        '''Gives a np.float64 value, requires values fot ALL variables'''
        values = [kwargs[v] for v in self.variables]
        return self.gradvalue(*values)

    def get_gradofx(self, **kwargs):
        '''Gives a the gradient as function of the inependent variale by substituting parameters for values.'''
        repl = [(s, kwargs[str(s)]) for s in self.symbols if str(s) in kwargs.keys()]
        self.gradofx = self.gradfunc.subs(repl)
        return self.gradofx


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


def pol3(x=1., k0=1., k1=1., k2=1., k3=1., **kwargs):
    '''Fit function, overload with keyword "l" to get string instead the callable function.'''
    expr = "k0 + x*k1 + x**2*k2 + x**3*k3"
    if 'l' in kwargs.keys():
        return expr.replace("np.", "")
    else:
        return eval(expr)

def logistics(x=1., A=1., w=0.1, x0=1., y0=1., **kwargs):
    '''Fit function, overload with keyword "l" to get string instead the callable function.'''
    expr = "A * (1 - 1/(1 + np.exp((x-x0)/w))) + y0"
    if 'l' in kwargs.keys():
         return expr.replace("np.", "")
    else:
        return eval(expr)
    
def arctan(x=1., A=-5., w=0.2, x0=1., y0=1., **kwargs):
    '''Fit function, overload with keyword "l" to get string instead the callable function.'''
    expr = "A * (np.arctan((x-x0)/w)/np.pi) + y0"#+1/2
    if 'l' in kwargs.keys():
        return expr.replace("np.", "").replace("arctan", "atan")
    else:
        return eval(expr)
    
def artanh(x=1., A=1., w=1., x0=1., y0=1., **kwargs):
    '''Fit function, overload with keyword "l" to get string instead the callable function.'''
    expr = "A * np.log((w/2+(x-x0))/(w/2-(x-x0))) + y0"
    if 'l' in kwargs.keys():
        return expr.replace("np.", "")
    else:
        return eval(expr)

def tan(x=1., A=-2., w=0.9, x0=0.8, y0=1.2, **kwargs):
    '''Fit function, overload with keyword "l" to get string instead the callable function.'''
    expr = "A * np.tan(np.pi*((x-x0)/w)) + y0"#-1/2
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
    if model.name == 'Model(artanh)':
        u = max(xdata)+0.001
        l = min(xdata)-0.001
        model.set_param_hint('W', value = u+l, min = u+l, vary = True)
        model.set_param_hint('x0', value = (u+l)/2, min = l, max = u, vary = True)
        model.set_param_hint('plus', expr = 'x0+W/2', value = u, min = u, vary = False)
        model.set_param_hint('minus', expr = 'x0-W/2', value = l, max = l, vary = False)
        model.make_params()
        model.print_param_hints()
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
    grad = sp.Matrix([sp.diff(function, s) for s in result.best_values.keys()])#symbs[1:]])
    M = sp.Matrix(result.covar.tolist())
    gTMg = grad.T * M * grad
    '''Debug'''
    #sp.pprint(function)
    #sp.pprint(grad)
    #sp.pprint(gTMg)
    best_values = [(s, result.best_values[str(s)]) for s in result.best_values.keys()]#symbs[1:]]
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
    xerr = np.array(len(x) * [0.])
    y = np.array(y)
    yerr = np.array(len(y) * [0.03])
    '''x valued for function plotting'''
    xvals = np.linspace(0.55, 1.15, 200)#x[0], x[-1], 100)
    '''Plot measurement'''    
    measurement = ax.errorbar(x, y, xerr = xerr, yerr = yerr, ls = '', label = yname.split(" / ")[0])
    '''Fit data'''
    fit = lmfit(model, x, y, yerr)
    text = ", ".join([r"$red. \chi^2 = %.2g$" % fit.redchi] + ["$%s = %.3g \pm %.1g$" % (key, fit.best_values[key], fit.stderrs[key]) for key in sorted(fit.best_values.keys())])
    fittext = plt.figtext(0., 1., text, fontsize = 8, color = measurement[0].get_color(), horizontalalignment= 'left')
    '''Plot fitted model'''
    function, = ax.plot(xvals, fit.best_function(xvals), c=measurement[0].get_color(), label = sp.latex(sp.sympify(model(l=True)), mode='inline', mul_symbol='dot'))
    '''Plot 1sigma intervall'''
    if fit.sigmaband:
        sigma = ax.fill_between(xvals, fit.best_function(xvals) + fit.sigmaband(xvals), fit.best_function(xvals) - fit.sigmaband(xvals), edgecolor = '', facecolor = measurement[0].get_color(), alpha = 0.1, label = r"$1\sigma$ Intervall")
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

    rfb = readlines(fname + ".csv")[:8]
    rfe = readlines(fname + ".csv")[8:]

    matplotlib_init()
    global f, ax
    f = plt.figure()
    ax = f.add_subplot(111)
    #f.suptitle(r"Capacitor Settings for Matched Resonance Circuits")

    fittext = []
    #fittext.append(fitdata(tan, rfb, [0, r"f / MHz", 1.], [1, r"$C_p$ / Skt", 1.]))
    #fittext.append(fitdata(arctan, rfb, [0, r"f / MHz", 1.], [2, r"$C_s$ / Skt", 1.]))
    fittext.append(fitdata(tan, rfe, [0, r"f / MHz", 1.], [1, r"$C_p$ / Skt", 1.]))
    fittext.append(fitdata(arctan, rfe, [0, r"f / MHz", 1.], [2, r"$C_s$ / Skt", 1.]))

    if len(fittext)*0.04-0.02 > 0.06:
        fittext_height = len(fittext)*0.04-0.02
    else:
        fittext_height = 0.06
    for i, text in enumerate(fittext):
        text.set_position((0.02, 1.-(i+1)*0.04))                 

    h, l = ax.get_legend_handles_labels()
    if len(l) == len(fittext)*3:
        '''Debug'''
        print(l)
        ax.legend((h[4], (h[0], h[2]), h[5], (h[1], h[3])), (l[4], l[0], l[5], l[1]), loc=1)
    else:
        ax.legend()
    ax.set_ylabel(r"$C /Skt$")
    #ax.set_xlim(0.55,1.1)
    ax.set_ylim(-0.5,5.5)
   
    plt.tight_layout(rect=(0,0,1,1-fittext_height))
    plt.draw()
    #plt.show()
    plt.savefig("." + os.sep + odir + oname + ".pdf")
    plt.savefig("." + os.sep + odir + oname + ".png")
    plt.close('all')
                
if __name__ == "__main__":
    main(sys.argv[1:])
