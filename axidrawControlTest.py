import math
#import noise
from pyaxidraw import axidraw   # import module
axi = axidraw.AxiDraw()          # Initialize class
axi.interactive()                # Enter interactive context
if not axi.connect():            # Open serial port to AxiDraw;
    print("not connected")
    quit()
print("connected!")
axi.options.units = 2


def circle(x, y, r):
    axi.moveto(x, y+r)
    steps = 100
    for i in range(steps + 1):
        theta = (i*2*math.pi)/steps
        axi.lineto(x + r*math.sin(theta), y + r*math.cos(theta))
    axi.penup()


def ellipse(x, y, rx, ry):
    axi.moveto(x, y+ry)
    steps = 100
    for i in range(steps + 1):
        theta = (i*2*math.pi)/steps
        axi.lineto(x + rx*math.sin(theta), y + ry*math.cos(theta))
    axi.penup()


def arc(x, y, rx, ry, thetaS, thetaE):
    if(thetaE < thetaS):
        print("Make starting angle less than ending angle")
        return
    
    axi.moveto(x + rx*math.sin(thetaS), y + ry*math.cos(thetaS))
    dTheta = thetaE-thetaS
    steps = int((dTheta/(2*math.pi))*100)
    for i in range(steps + 1):
        theta = thetaS + (i*dTheta)/steps
        axi.lineto(x + rx*math.sin(theta), y + ry*math.cos(theta))
    axi.penup()


def rect(x, y, w, h):
    axi.moveto(x, y)
    axi.lineto(x+w, y)
    axi.lineto(x+w, y+h)
    axi.lineto(x, y+h)
    axi.lineto(x, y)
    axi.penup()


axi.moveto(0, 0)

#rect(20, 20, 40, 70)
#ellipse(100, 70, 30, 60)
arc(50, 50, 30, 40, 0, math.pi)


axi.moveto(0, 0)
axi.disconnect() 