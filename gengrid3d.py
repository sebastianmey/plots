#!/usr/bin/python
import getopt, sys, os


def usage():
    """
    Usage function
    """
    print("""Usage: %s -o <output-file> -r [x-range,y-range,z-range] -m [multiplier]

-h   show this help message and exit
-r   give INTEGER dimensions of the created grid symmetric around 0, separated by \",\"
-m   multiplier to determine unit of length
""" %sys.argv[0])

    
def symGrid3D(xrange, yrange, zrange, mult):
    """
    Generate 3D Grid symmetric around (0, 0, 0)
    """
    grid = []
    for x in range(-xrange, xrange+1):
        for y in range(-yrange, yrange+1):
            for z in range(-zrange, zrange+1):
                grid.append([float(x)*mult, float(y)*mult, float(z)*mult])
    return grid



def main(argv):
    ofile = 0
    x = 0
    y = 0
    z = 0
    m = 0
    try:                                
        opts, args = getopt.getopt(argv, "ho:r:m:")
        if not "-o" and "-r" and "-m" in opts:
            raise Exception("insufficient CMD arguments")
    except (getopt.GetoptError, Exception) as err:
        print(str(err) + "\n")
        usage()                      
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            usage()
            sys.exit()
        elif opt == "-o":
            ofile = arg        
        elif opt == "-r":
            x = int(arg.split(",")[0])
            y = int(arg.split(",")[1])
            z = int(arg.split(",")[2])
        elif opt == "-m":
            m = float(arg)
    grid3d = symGrid3D(x, y, z, m)
    f = open(ofile, 'w')
    for line in grid3d:
        f.write(" ".join(map(str, line)) + "\n")
    f.close

if __name__ == "__main__":
    main(sys.argv[1:])
