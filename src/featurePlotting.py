import math
import numpy as np
from pyaxidraw import axidraw
import random as rand



#Basic geometrical functions

def rect(axi, x1, y1, x2, y2):
    axi.moveto(x1, y1)
    axi.lineto(x2, y1)
    axi.lineto(x2, y2)
    axi.lineto(x1, y2)
    axi.lineto(x1, y1)
    axi.penup()


def circle(axi, x, y, r):
    axi.moveto(x, y+r)
    steps = 16
    for i in range(steps + 1):
        theta = (i*2*math.pi)/steps
        axi.lineto(x + r*math.sin(theta), y + r*math.cos(theta))
    axi.penup()


def ellipse(axi, x, y, rx, ry):
    axi.moveto(x, y+ry)
    steps = 16
    for i in range(steps + 1):
        theta = (i*2*math.pi)/steps
        axi.lineto(x + rx*math.sin(theta), y + ry*math.cos(theta))
    axi.penup()


def arc(axi, x, y, rx, ry, thetaS, thetaE, raisePen):
    if raisePen:
        axi.penup()

    if(thetaE < thetaS):
        print("Make starting angle less than ending angle")
        return
    
    axi.goto(x + rx*math.sin(thetaS), y + ry*math.cos(thetaS))
    dTheta = thetaE-thetaS
    steps = int((dTheta/(2*math.pi))*50)
    for i in range(steps + 1):
        theta = thetaS + (i*dTheta)/steps
        axi.lineto(x + rx*math.sin(theta), y + ry*math.cos(theta))

    if raisePen:
        axi.penup()


def convertDrawPoly(axi, pointsList, S, xc, yc):
    axi.moveto(pointsList[0]*S+xc, pointsList[1]*S+yc)

    for i in range(2, len(pointsList), 2):
        axi.lineto(pointsList[i]*S+xc, pointsList[i+1]*S+yc)

    axi.lineto(pointsList[0]*S+xc, pointsList[1]*S+yc)
    axi.penup()




#convert from PIL drawing convention to axi drawing convention

def convertArc(x1, y1, x2, y2, r1, r2, S, xc, yc):
    x1 = x1*S + xc
    y1 = y1*S + yc
    x2 = x2*S + xc
    y2 = y2*S + yc

    thetaS = math.radians(r1-90)
    thetaE = math.radians(r2-90)

    rx = (x2-x1)/2
    ry = (y2-y1)/2
    x = x1+rx
    y = y1+ry

    return [x, y, rx, ry, thetaS, thetaE]




#Functions for drawing doodles

def bird(axi, desc, S, xc, yc):
    [x, y, birdScale, bend] = desc
    bend = math.radians(bend)
    x = x*S+xc
    y = y*S+yc
    birdScale*=S
    offset = 2 * birdScale * np.cos(np.pi/2 - bend)

    axi.penup()
    arc(axi, x+offset, y, birdScale, birdScale, np.pi-bend, np.pi+bend, False)
    arc(axi, x, y, birdScale, birdScale, np.pi-bend, np.pi+bend, False)
    #arc(x*S+xc + np.cos((np.pi/2)-y)*birdScale*S*6, y, birdScale, birdScale, np.pi-cutoff, np.pi+cutoff, False)
    axi.penup()



def tree(axi, desc, S, xc, yc):
    [trunk, r] = desc

    #convert to axi coordinates
    trunk = [i * S for i in trunk]
    r*=S

    trunk[0] += xc
    trunk[1] += yc
    trunk[2] += xc
    trunk[3] += yc
    
    #Trunk
    axi.goto(trunk[0], trunk[1])
    axi.lineto(trunk[2], trunk[3])

    #Crown
    circle(axi, trunk[2], trunk[3], r)

    axi.penup()



def striation(axi, endpoints, S, xc, yc):
    #convert to axi coordinates
    endpoints = [i * S for i in endpoints]

    endpoints[0] += xc
    endpoints[1] += yc
    endpoints[2] += xc
    endpoints[3] += yc

    axi.goto(endpoints[0], endpoints[1])
    axi.lineto(endpoints[2], endpoints[3])
    axi.penup()



def lake(axi, desc, S, xc, yc):
    [waveInfo, boatInfo, fishInfo] = desc
    [startX, endX, y, waveD, waveNum] = waveInfo

    #convert to axi coordinates
    startX = startX*S+xc
    endX = endX*S+xc
    y = y*S+yc
    waveD*=S

    #Draw lake
    axi.moveto(startX, y)
    axi.lineto(startX+waveD, y)

    for i in range(waveNum-2):
        arc(axi, startX+waveD*i+waveD*1.5, y, waveD/2, waveD/2, np.pi*1.5, np.pi*2.5, False)

    axi.moveto(endX-waveD, y)
    axi.lineto(endX, y)


    #Draw boat
    if boatInfo != -1:
        [boatX, boatScale, sailDir] = boatInfo
        boatScale*=S
        boatX = boatX*S+xc
        boatY = y - boatScale/4
        
        arc(axi, boatX, boatY, boatScale, boatScale/2, np.pi*1.5, np.pi*2.5, True) 
        axi.moveto(boatX-boatScale, boatY)
        axi.lineto(boatX+boatScale, boatY)

        if sailDir:
            axi.moveto(boatX-boatScale/3, boatY)
            axi.lineto(boatX-boatScale/3, boatY-(2*boatScale))
            axi.lineto(boatX+(2/3)*boatScale, boatY)
            axi.lineto(boatX-boatScale/3, boatY)
        else:
            axi.moveto(boatX+boatScale/3, boatY)
            axi.lineto(boatX+boatScale/3, boatY-(2*boatScale))
            axi.lineto(boatX-(2/3)*boatScale, boatY)
            axi.lineto(boatX+boatScale/3, boatY)


    #Draw fish
    for fish in fishInfo:
        [upperBody, lowerBody, tail] = fish
        upperBodyArgs = convertArc(*upperBody, S, xc, yc)
        lowerBodyArgs = convertArc(*lowerBody, S, xc, yc)

        axi.penup()
        arc(axi, *upperBodyArgs, False)
        arc(axi, *lowerBodyArgs, False)
        convertDrawPoly(axi, tail, S, xc, yc)

        
    axi.penup()



def tower(axi, desc, S, xc, yc):
    axi.penup()

    [leftWall, rightWall, roof] = desc  #get each drawn component of the feature

    #convert to axi coordinates
    leftWall = [i * S for i in leftWall]
    rightWall = [i * S for i in rightWall]
    roof = [i * S for i in roof]

    leftWall[0]+=xc
    leftWall[1]+=yc
    leftWall[2]+=xc
    leftWall[3]+=yc

    rightWall[0]+=xc
    rightWall[1]+=yc
    rightWall[2]+=xc
    rightWall[3]+=yc

    for i in range(0, len(roof), 2):
        roof[i]+=xc
        roof[i+1]+=yc

    #draw walls
    axi.moveto(leftWall[0], leftWall[1])
    axi.lineto(leftWall[2], leftWall[3])
    axi.moveto(rightWall[0], rightWall[1])
    axi.lineto(rightWall[2], rightWall[3])

    #draw roof
    if len(roof) == 4:  #rectangular roof
        rect(axi, roof[0], roof[1], roof[2], roof[3])
    elif len(roof) == 6:  #triangular roof
        axi.moveto(roof[0], roof[1])
        axi.lineto(roof[2], roof[3])
        axi.lineto(roof[4], roof[5])
        axi.lineto(roof[0], roof[1])
    else:  #roof error
        print("Error: invalid roof array length: ", len(roof))
    
    axi.penup()