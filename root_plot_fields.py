#!/usr/bin/python
import getopt, sys, os, subprocess, math, numpy, ROOT


def usage():
    """Usage function"""
    print("""Usage: %s -B <B-file> -E <E-file> -l -o <output-file>

-l   line with data caption in input-file
-n   normalizing factor for fields (e.g. current)
-p   plot fields in ROOT
-h   show this help message and exit
""" %sys.argv[0])


def readFrom(file, start):
    """Read in lines as list of strings, cut header"""
    f = open(file, 'r')
    lines = []
    try:
        for line in f:
            lines.append(line.split())
    finally:
        f.close
        del lines[0:(start-1)]
        return lines


def wien(betaZ, Bfield, Efield):
    """Calculate Lorentz Force and Impedance out of E and B field"""
    print("Calculating LORENTZ-condition...")
    wien = [Bfield[0][0:6] + Efield[0][3:6] + ["FBx(eV/m)", "FBy(eV/m)", "FBz(eV/m)"] + ["FEx(eV/m)", "FEy(eV/m)", "FEz(eV/m)"] + ["Fx(eV/m)", "Fy(eV/m)", "Fz(eV/m)", "Z(Ohm)"]]
    q=1
    c=299792458
    for b,e in zip(Bfield[1:], Efield[1:]):
        line = b[0:6] + e[3:6] + [q*(-betaZ*float(b[4])*c), q*(betaZ*float(b[3])*c), 0] + [q*float(e[3]), q*float(e[4]), q*float(e[5])] + [q*(float(e[3])-betaZ*float(b[4])*c), q*(float(e[4])+betaZ*float(b[3])*c), q*float(e[5]), -4e-7*math.pi*float(e[4])/(float(b[3]))]
        wien.append(line)
    return wien


def sort4d(data, column):
    dict = {}
    """set generates a unsorted list of all unique values in column of data"""
    for key in set(l[column] for l in data):
        for line in data:
            if key == line[column]:
                """if dictionary entry exists, append line, else create new entry"""
                try:
                    dict[float(key)].append(line)
                except KeyError:
                    dict[float(key)] = [line]
    return dict


def plot3d(data, column1, xname, column2, yname, column3, zname, zmult):
    """3-dimesional plot of columns in list "data" with ROOT"""
    print("Plotting...")
    graph = ROOT.TGraph2D()
    i=0
    for point in data:
        graph.SetPoint(i, float(point[column1]), float(point[column2]), float(zmult)*float(point[column3]))
        i = i + 1
    graph.GetHistogram()
    """for fieldsumplots"""
    graph.GetHistogram().SetMaximum(210000.)
    graph.GetHistogram().SetMinimum(-220000.)
    """ """
    graph.SetTitle()
    for axis in [graph.GetXaxis(), graph.GetYaxis(), graph.GetZaxis()]:
        axis.SetNdivisions(505)
        axis.SetLabelSize(0.07)
        axis.SetLabelFont(42)
        axis.SetTitleSize(0.07)
        axis.SetTitleFont(42)
        axis.CenterTitle(1)
    graph.GetXaxis().SetTitleOffset(1.8)
    graph.GetXaxis().SetTitle("%s / %s" % (xname.split("(")[0], xname.split("(")[1][:-1]))
    graph.GetYaxis().SetTitleOffset(1.1)
    graph.GetYaxis().SetTitle("%s / %s" % (yname.split("(")[0], yname.split("(")[1][:-1]))
    graph.GetZaxis().SetTitleOffset(1.)
    graph.GetZaxis().SetTitle("Fy / eV/m")#("%s / %s" % (zname.split("(")[0], zname.split("(")[1][:-1]))
    graph.SetFillStyle(4000)
    return graph


def main(argv):
    """read in CMD arguments"""
    gdfpath = os.path.expanduser("~mey") + os.sep + 'bin' + os.sep + 'gpt310x64' + os.sep + 'bin' + os.sep
    headerlength = 1
    norm = 1.0
    Bfile = ""
    Efile = ""
    ofile = ""
    plot = 0
    try:                                
        opts, args = getopt.getopt(argv, "hB:E:o:l:n:p")
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
        elif opt == "-o":
            ofile = arg
        elif opt == "-l":
            headerlength = int(arg)
        elif opt == "-n":
            norm = float(arg)
        elif opt == "-p":
            plot = 1

    if Bfile:
        B = readFrom(Bfile, headerlength)
        field3d = B
    if Efile:
        E = readFrom(Efile, headerlength)
        field3d = E
    if Bfile and Efile:
        field3d = wien(0.459, B, E)

    """sort 4d-file by y axis entries into dictionary"""
    field2d = sort4d(field3d[1:], 1)

    """sorted data output"""
    if ofile:
        g = open(ofile, 'w')
        g.write("\t".join(map(str, field3d[0])) + "\n")
        for key in sorted(field2d.keys()):
            for line in field2d[key]:
                g.write("\t".join(map(str, line)) + "\n")
        g.close()
#        gdf=subprocess.Popen([gdfpath + 'asci2gdf', '-o %s' % ofile[:-4] + '.gdf', ofile])
                
    """Plot data"""
    if plot:
        sys.argv = []
        ROOT.gStyle.SetNumberContours(99)
        i = 0
        j = 0
        for key in sorted(field2d.keys()):
            """skip everything except y=0 plane"""
            if float(key) != 0.0: 
                continue
            for j in xrange(3,len(field3d[0])):
                """only fields have to be normalized, not the Lorentz-Force"""
                if j >= 9:
                    norm = 1.0
                """ select what to plot"""
                if j != 10:# in xrange(9,15):#!= 16:#for fieldsumplots
                    continue
#                plot = plot3d(field2d[key], 0, field3d[0][0], 2, field3d[0][2], j, field3d[0][j], 1/norm)   
                plotFB = plot3d(field2d[key], 0, field3d[0][0], 2, field3d[0][2], j, field3d[0][j], 1/norm)
                plotFE = plot3d(field2d[key], 0, field3d[0][0], 2, field3d[0][2], j+3, field3d[0][j+3], 1/norm)
                canvas = ROOT.TCanvas("%s_%s" % (field3d[0][j], str(i)), "%s in x-z-plane at y = %s m" % (field3d[0][j], key), 1920, 1080)
                canvas.SetTheta(15.)
                canvas.SetPhi(275.)
                canvas.SetLeftMargin(0.16)
                canvas.SetRightMargin(0.13)
                canvas.SetBottomMargin(0.16)
#                plot.SetTitle("%s in x-z-plane at y = %s m" % (field3d[0][j], key))
#                plot.Draw('SURF2Z')
#                plot.Draw('Cont1 SAME')
                plotFB.Draw('SURF2')
#                plotFB.Draw('CONT1 SAME')
                plotFE.Draw('SURF2Z SAME')
#                plotFE.Draw('CONT1 SAME')              
                raw_input("""Press ENTER to continue.""")
                canvas.Print("data/%s_y%s_%s" % (field3d[0][j].split("(")[0], key, str(i)) + ".pdf")
#                canvas.Print("data/%s_y%s_%s" % (field3d[0][j].split("(")[0], key, str(i)) + ".png")
#                canvas.SaveAs("data/%s_y%s_%s" % (field3d[0][j].split("(")[0], key, str(i)) + ".root")
                j = j+1
            i = i+1
        raw_input("""Press ENTER to finish.""")


if __name__ == "__main__":
    main(sys.argv[1:])
