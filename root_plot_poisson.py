#!/usr/bin/python
import getopt, sys, os, subprocess, math, ROOT

def usage():
    """Usage function"""
    print("""AUTOMESH,POISSON,WSFPLOT, put out Grid file and plot fields and Lorentz Force in ROOT

Usage: %s -B <B-field .am file> -E <E-field .am file> -g [\"x_start y_start, x_end y_end\"] -n -p

-h    Show this help message and exit
-g    interpolation grid start and end coordinates in units of mm. Please mind the quotation marks!
-n    no calculation, use files in directory
-p    3D plot of fields and Lorentz-Force with ROOT
""" % sys.argv[0])

def readFrom(file, start):
    """Read in lines as list of strings, cut header"""
    f = open(file, 'r')
    lines = []
    try:
        for line in f:
            lines.append(line.split())
    finally:
        del lines[0:(start-1)]
        f.close
        return lines
    
def poisson(progpath, basename):
    """Automesh, Poisson and fieldline plot on file basename."""
    # Automesh
    print("Calculating mesh...")
    mesh = subprocess.Popen(['wine', progpath + "AUTOMESH.EXE", basename + ".am"])
    mesh.wait()
    # Poisson
    print("Calculating fields...")
    fields = subprocess.Popen(['wine', progpath + "POISSON.EXE", basename + ".T35"])
    fields.wait()
    # Plot fieldlines
    subprocess.Popen(['wine', progpath + "WSFPLOT.EXE", basename + ".T35"])
    
def grid(progpath, basename, start, end):
    """Generate .in7 File and run SF7"""
    steps = [abs(float(start.split()[0]) - float(end.split()[0]))/10, abs(float(start.split()[1]) - float(end.split()[1]))/10]
    # Write gridfile
    in7 = open(basename + ".in7", 'w')
    in7.write("""Grid
%s %s
%s %s
END""" %(start, end, steps[0], steps[1]))
    in7.close()
    # SF7
    print("Interpolating fields...")
    sf7 = subprocess.Popen(['wine', progpath + "SF7.EXE", basename + ".in7", basename + ".T35"])
    sf7.wait()
    os.rename("OUTSF7.TXT",  basename + "_GRID.TXT")

def wien(betaZ, B, E):
    """Calculate Lorentz Force and Impedance out of E and B field and print it to file"""
    print("Calculating LORENTZ-condition...")
    f = open("WIEN_DC_F_GRID.TXT",'w')
    f.write("x/mm \t y/mm \t Fx/e / V/m \t Fy/e / V/m \t Z=mu*Ey/Bx /Ohm \n")
    wien = []
    q=1.60217653e-19
    c=299792458
    for b,e in zip(B,E):
        line = [float(b[0]), float(b[1]), float(e[2])*100-betaZ*float(b[3])*c/10000, float(e[3])*100+betaZ*float(b[2])*c/10000, 4e-7*math.pi*float(e[3])*100/(float(b[2])/10000)]
        wien.append(line)
        f.write("\t".join(map(str, line)) + "\n")
    f.close
    return wien
                                                    
def plot3d(data, column1, xname, column2, yname, column3, zname, zmult):
    """3-dimesional plot of columns in list "data" with ROOT"""
    print("Plotting...")
    graph = ROOT.TGraph2D()
    i=0
    for point in data:
        graph.SetPoint(i, float(point[column1]), float(point[column2]), float(zmult)*float(point[column3]))
        i = i + 1
    graph.SetTitle()
    graph.GetXaxis().CenterTitle(1)
    graph.GetXaxis().SetTitle("%s" %xname)
    graph.GetXaxis().SetTitleOffset(1.5)
    graph.GetYaxis().CenterTitle(1)
    graph.GetYaxis().SetTitle("%s" %yname)
    graph.GetYaxis().SetTitleOffset(1.5)
    graph.GetZaxis().CenterTitle(1)
    graph.GetZaxis().SetTitle("%s" %zname)
    graph.GetZaxis().SetTitleOffset(1.5)
    return graph

def main(argv):
    # Superfish drirectory
    fishpath = os.path.expanduser("~mey") + os.sep + 'bin' + os.sep + 'superfish' + os.sep
    fnameB = 0
    fnameE = 0
    gstart = 0
    gend = 0
    calculate = 1
    plot = 0
    # Read CMD-arguments given
    try:                                
        opts, args = getopt.getopt(argv, "hB:E:g:np")
    except getopt.GetoptError as err:
        print(str(err)+"\n")
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            usage()
            sys.exit()
        elif opt == "-B":
            fnameB = arg[:-3]
        elif opt == "-E":
            fnameE = arg[:-3]
        elif opt == "-g":
            gstart = arg.split(", ")[0]
            gend = arg.split(", ")[1]
        elif opt == "-n":
            calculate = 0
        elif opt == "-p":
            plot = 1
    # Calculate fields
    if calculate:
        if fnameB:
            poisson(fishpath, fnameB)
        if fnameE:
            poisson(fishpath, fnameE)
    # Interpolate fields
    if gstart and gend:
        if fnameB:
            grid(fishpath, fnameB, gstart, gend)
            fieldB = readFrom(fnameB + "_GRID.TXT", 35)
        if fnameE:
            grid(fishpath, fnameE, gstart, gend)
            fieldE = readFrom(fnameE + "_GRID.TXT", 35)
        # Calculate LORENTZ condition
        if fnameB and fnameE:
            force = wien(0.459, fieldB, fieldE)
    # Plot
    if plot:
        sys.argv = []
        ROOT.gStyle.SetNumberContours(50)
        if fnameB:
            plotBx = plot3d(fieldB, 0, "x / mm", 1, "y / mm", 2,  "Bx / mT", 0.1)
            Bx = ROOT.TCanvas('Bx', 'Bx in x-y-plane')
            plotBx.Draw('SURF1Z')
            Bx.Print("%s" % fnameB + "x.pdf")
            Bx.SaveAs("%s" % fnameB + "x.root")

            plotBy = plot3d(fieldB, 0, "x / mm", 1, "y / mm", 3,  "By / mT", 0.1)
            By = ROOT.TCanvas('By', 'By in r-z-plane')
            plotBy.Draw('SURF1Z')
            By.Print("%s" % fnameB + "y.pdf")
            By.SaveAs("%s" % fnameB + "y.root")

        if fnameE:
            plotEx = plot3d(fieldE, 0, "x / mm", 1, "y / mm", 2,  "Ex / V/m", 100)
            Ex = ROOT.TCanvas('Ex', 'Ex in x-y-plane')
            plotEx.Draw('SURF2Z')
            Ex.Print("%s" % fnameE + "x.pdf")
            Ex.SaveAs("%s" % fnameE + "x.root")

            plotEy = plot3d(fieldE, 0, "x / mm", 1, "y / mm", 3,  "Ey / V/m", 100)
            Ey = ROOT.TCanvas('Ey', 'Ey in x-y-plane')
            plotEy.Draw('SURF2Z')
            Ey.Print("%s" % fnameE + "y.pdf")
            Ey.SaveAs("%s" % fnameE + "y.root")

        if fnameB and fnameE:
            plotFx = plot3d(force, 0, "x / mm", 1, "y / mm", 2, "Fx/e / V/m", 1)
            Fx = ROOT.TCanvas('Fx', 'Fx/e in x-y-plane')
            plotFx.Draw('SURF2Z')
            Fx.Print("WIEN_DC_Fx.pdf")
            Fx.SaveAs("WIEN_DC_Fx.root")

            plotFy = plot3d(force, 0, "x / mm", 1, "y / mm", 3, "Fy/e / V/m", 1)
            Fy = ROOT.TCanvas('Fy', 'Fy/e in x-y-plane')
            plotFy.Draw('SURF2Z')
            Fy.Print("WIEN_DC_Fy.pdf")        
            Fy.SaveAs("WIEN_DC_Fy.root")        

            plotZ = plot3d(force, 0, "x / mm", 1, "y / mm", 4, "Z / Ohm", 1)
            Z = ROOT.TCanvas('Z', 'Z=mu Ey/Bx in x-y-plane')
            plotZ.Draw('SURF2Z')
            Z.Print("WIEN_DC_Z.pdf")
            Z.SaveAs("WIEN_DC_Z.root")

        raw_input("""Press ENTER to finish.""")
            
if __name__ == "__main__":
    main(sys.argv[1:])
