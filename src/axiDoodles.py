import math
import numpy as np
from pyaxidraw import axidraw
import random as rand



#Basic geometrical functions

def rect(axi, x, y, w, h):
    axi.moveto(x, y)
    axi.lineto(x+w, y)
    axi.lineto(x+w, y+h)
    axi.lineto(x, y+h)
    axi.lineto(x, y)
    axi.penup()


def circle(axi, x, y, r):
    axi.moveto(x, y+r)
    steps = 30
    for i in range(steps + 1):
        theta = (i*2*math.pi)/steps
        axi.lineto(x + r*math.sin(theta), y + r*math.cos(theta))
    axi.penup()


def ellipse(axi, x, y, rx, ry):
    axi.moveto(x, y+ry)
    steps = 30
    for i in range(steps + 1):
        theta = (i*2*math.pi)/steps
        axi.lineto(x + rx*math.sin(theta), y + ry*math.cos(theta))
    axi.penup()


def arc(axi, x, y, rx, ry, thetaS, thetaE, raisePen):
    if(thetaE < thetaS):
        print("Make starting angle less than ending angle")
        return
    
    steps = int(30*(((thetaE-thetaS)/(2*np.pi))))
    
    if raisePen:
        axi.penup()

    axi.goto(x + rx*math.sin(thetaS), y + ry*math.cos(thetaS))
    dTheta = thetaE-thetaS
    steps = int((dTheta/(2*math.pi))*100)
    for i in range(steps + 1):
        theta = thetaS + (i*dTheta)/steps
        axi.lineto(x + rx*math.sin(theta), y + ry*math.cos(theta))
    if raisePen:
        axi.penup()



#Functions for drawing doodles


def drawBird(axi, x, y, birdScale, xc, yc, S):
    cutoff = 1
    axi.moveto(x*S+xc, y)
    arc(x*S+xc, y, birdScale, birdScale, np.pi-cutoff, np.pi+cutoff, True)
    arc(x*S+xc + np.cos((np.pi/2)-cutoff)*birdScale*S*6, y, birdScale, birdScale, np.pi-cutoff, np.pi+cutoff, True)

    axi.penup()



def drawTree(axi, i, x, y, maxh):
    h = rand.uniform(-3, maxh)  #height of trunk
    axi.moveto(x, y)
    axi.lineto(x, y+h)
    r = rand.uniform(h/7, h/2)  #radius of crown
    circle(x, y + h, r)  #draw crown
    if r > maxh/1.5 and rand.random() < 0.2:  #if crown is big add branches
        axi.moveto(x, (y+h)-rand.uniform(r/2.2, r/8))
        axi.lineto(x-r/6,  y+h-rand.uniform(-r/8, r/8))
        axi.moveto(x, (y+h)-rand.uniform(r/2.2, r/8))
        axi.lineto(x+r/6,  y+h-rand.uniform(-r/8, r/8))
        axi.moveto(x, y+h)
        axi.lineto(x, y+h+r/8)

    elif r > maxh/1.5 and rand.random() < 0.2:
        arc(x, y+h, r/2, r/2, np.pi/2, 3*(np.pi)/2, True)
        axi.moveto(x, y+h)
        axi.lineto(x, y+h+r/8)

    axi.penup()
    return i+int(r)+1



def drawStriation(axi, tracedLine, i, x, y, minStria, maxStria, S, xc, yc):
    isStria = True
    for c in range(minStria):
        if i+c >= len(tracedLine):
            isStria = False
            break

        if y < tracedLine[i+c][1]:
            isStria = False
    
    rX = -1
    if isStria:
        for c in range(minStria, maxStria):
            if i+c >= len(tracedLine):
                break

            if y < tracedLine[i+c][1]:
                rX = i + c
                break
    
    if rX != -1:
        print("Drawing: Striation")
        axi.goto(x*S+xc, y*S+yc)
        axi.lineto(tracedLine[rX][0]*S+xc, tracedLine[rX][1]*S+yc)

    axi.penup()



def drawBoats(axi, lineArr, startIdx, x, y, waveLen, endIdx, xc, yc, S):
    lakeW = (endIdx - startIdx)*S
    waveNum = int(lakeW/waveLen)
    leftOver = lakeW%waveLen
    waveD = waveLen + leftOver/waveNum
    #vsk.line(x, y, x+waveD, y)
    axi.moveto(x*S+xc, y*S+yc)
    axi.lineto(x*S+xc+waveD, y*S+yc)
    boatloc = int(rand.randrange(0, int((waveNum-2)*2)))
    for i in range(waveNum-2):
        arc(x*S+xc+waveD*i+waveD*1.5, (lineArr[endIdx][1]*(i/waveNum) + y*((waveNum-i)/waveNum))*S+yc, waveD/2, waveD/2, np.pi*1.5, np.pi*2.5, False)
        if i == boatloc:  #generate boats
            print("Drawing: Boat")
            boatScale = (6 + waveNum/7 )*S
            boatX = x*S+waveD*i+waveD*1.5 + xc
            boatY = (lineArr[endIdx][1]*(i/waveNum) + y*((waveNum-i)/waveNum))*S - boatScale/4 + yc
            sailDir = rand.random() < 0.5
            
            arc(boatX, boatY, boatScale, boatScale/2, np.pi*1.5, np.pi*2.5, True) 
            #vsk.line(boatX-boatScale, boatY, boatX+boatScale, boatY)
            axi.moveto(boatX-boatScale, boatY)
            axi.lineto(boatX+boatScale, boatY)
            if sailDir:
                #vsk.triangle(boatX-boatScale/3, boatY, boatX-boatScale/3, boatY-(2*boatScale), boatX+(2/3)*boatScale, boatY)
                axi.moveto(boatX-boatScale/3, boatY)
                axi.lineto(boatX-boatScale/3, boatY-(2*boatScale))
                axi.lineto(boatX+(2/3)*boatScale, boatY)
                axi.lineto(boatX-boatScale/3, boatY)
            else:
                #vsk.triangle(boatX+boatScale/3, boatY, boatX+boatScale/3, boatY-(2*boatScale), boatX-(2/3)*boatScale, boatY)
                axi.moveto(boatX+boatScale/3, boatY)
                axi.lineto(boatX+boatScale/3, boatY-(2*boatScale))
                axi.lineto(boatX-(2/3)*boatScale, boatY)
                axi.lineto(boatX+boatScale/3, boatY)

            axi.penup()

    #vsk.line(lineArr[endIdx][0], lineArr[endIdx][1], lineArr[endIdx][0] - waveD, lineArr[endIdx][1])
    axi.moveto(lineArr[endIdx][0]*S+xc - waveD, lineArr[endIdx][1]*S+yc)
    axi.lineto(lineArr[endIdx][0]*S+xc, lineArr[endIdx][1]*S+yc)
    axi.penup()



def drawLake(axi, tracedLine, i, x, y, minLake, maxLake, waveLen, S, xc, yc):
    isStria = True
    for c in range(minLake):
        if i+c >= len(tracedLine):
            isStria = False
            break

        if y > tracedLine[i+c][1]:
            isStria = False
    
    endIdx = 0
    if isStria:
        for c in range(minLake, maxLake):
            if i+c >= len(tracedLine):
                break

            if y > tracedLine[i+c][1]:
                endIdx = i + c
                break
    
    if endIdx != 0:
        print("Drawing: Lake")
        drawBoats(tracedLine, i, x, y, waveLen, endIdx, xc, yc, S)

    axi.penup()
    return endIdx



def drawTower(axi, tracedLine, i, S, xc, yc):
    #find left and right wall of tower
    l = i - rand.randrange(1, 4)
    r = i + rand.randrange(1, 4)

    #make sure tower sides are in bounds of tracedLine
    l = max(0, l)
    r = min(len(tracedLine) - 1, r)

    #determines dimensions of tower
    lx = tracedLine[l][0]*S + xc
    ly = tracedLine[l][1]*S + yc
    rx = tracedLine[r][0]*S + xc
    ry = tracedLine[r][1]*S + yc
    h = (tracedLine[r][1]*S + yc) - rand.uniform(6, 30)*S
    w = rx-lx

    axi.moveto(lx, ly)
    axi.lineto(lx, h)
    axi.moveto(rx, ry)
    axi.lineto(rx, h)

    wt = w * rand.uniform(1.2, 2.2)
    d = i + int(wt/2)+1
    if rand.random() < 0.3:
        ht = wt*rand.uniform(0.15, 0.25)
        rect((lx+rx)/2 - wt/2, h-ht, wt, ht)
    else:
        ht = min((ry - h)/1.5, w*rand.uniform(0.7, 3))
        axi.moveto((lx+rx)/2 - wt/2, h)
        axi.lineto((lx+rx)/2 + wt/2, h)
        axi.lineto((lx+rx)/2, h - ht)
        axi.lineto((lx+rx)/2 - wt/2, h)
    
    axi.penup()
    return d


def drawVillage(axi, tracedLine, i, S, xc, yc):
        c = 0
        while abs(tracedLine[i+c][1] - tracedLine[i+c+1][1]) < 0.5:
            print("Drawing: Village House")
            w = rand.randint(3, 10)
            if i + c + 8 >= len(tracedLine):
                break

            hb = tracedLine[i+c][1]*S+yc - rand.uniform(4, 12)*S

            lx = tracedLine[i+c][0]*S+xc
            ly = tracedLine[i+c][1]*S+yc 
            rx = tracedLine[i+c+w][0]*S+xc
            ry = tracedLine[i+c+w][1]*S+yc
            #vsk.line(lx, tracedLine[i+c][1], lx, h)
            axi.moveto(lx, ly)
            axi.lineto(lx, hb)
            #vsk.line(rx, tracedLine[i+c+w][1], rx, h)
            axi.moveto(rx, ry)
            axi.lineto(rx, hb)

            wt = w*rand.uniform(1, 1.4)
            ht = wt*rand.uniform(0.4, 0.7)

            #vsk.triangle((lx+rx)/2 - wt/2, h, (lx+rx)/2 + wt/2, h, (lx+rx)/2, h - ht)
            axi.moveto((lx+rx)/2 - (wt/2)*S, hb)
            axi.lineto((lx+rx)/2 + (wt/2)*S, hb)
            axi.lineto((lx+rx)/2, hb - ht*S)
            axi.lineto((lx+rx)/2 - (wt/2)*S, hb)

            c+=rand.randint(1, 3)
        
        axi.penup()
        d = i+c+1
        return d