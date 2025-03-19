import cv2 as cv
import numpy as np
import random as rand
import statistics as stat
import math
from pyaxidraw import axidraw  
from scipy.interpolate import splprep, splev
import keyboard
import argparse
import imageConverter as imgc
from imageConverter import Island
import threading


def circle(x, y, r):
    axi.moveto(x, y+r)
    steps = 30
    for i in range(steps + 1):
        theta = (i*2*math.pi)/steps
        axi.lineto(x + r*math.sin(theta), y + r*math.cos(theta))
    axi.penup()


def ellipse(x, y, rx, ry):
    axi.moveto(x, y+ry)
    steps = 30
    for i in range(steps + 1):
        theta = (i*2*math.pi)/steps
        axi.lineto(x + rx*math.sin(theta), y + ry*math.cos(theta))
    axi.penup()


def arc(x, y, rx, ry, thetaS, thetaE, raisePen):
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


def rect(x, y, w, h):
    axi.moveto(x, y)
    axi.lineto(x+w, y)
    axi.lineto(x+w, y+h)
    axi.lineto(x, y+h)
    axi.lineto(x, y)
    axi.penup()



# Trackbars update live
def on_thresh(val):
    global thresh
    thresh = val

def on_xMin(val):
    global cropXmin
    cropXmin = val

def on_yMin(val):
    global cropYmin
    cropYmin = val

def on_cropW(val):
    global cropW
    cropW = val

def on_cropH(val):
    global cropH
    cropH = val



def on_xstrackbar(val):
    global xc
    xc = (val*-1)/10

def on_ystrackbar(val):
    global yc
    yc = (val*-1)/10

def on_Strackbar(val):
    global S
    S = val/1000





#Functions for drawing doodles

def drawStriation(tracedLine, i, x, y, minStria, maxStria, S, xc, yc):
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


def drawBoats(lineArr, startIdx, x, y, waveLen, endIdx, xc, yc, S):
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


def drawLake(tracedLine, i, x, y, minLake, maxLake, waveLen, S, xc, yc):
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


def drawBird(x, y, birdScale, xc, yc, S):
    cutoff = 1
    axi.moveto(x*S+xc, y)
    arc(x*S+xc, y, birdScale, birdScale, np.pi-cutoff, np.pi+cutoff, True)
    arc(x*S+xc + np.cos((np.pi/2)-cutoff)*birdScale*S*6, y, birdScale, birdScale, np.pi-cutoff, np.pi+cutoff, True)

    axi.penup()


def drawTree(i, x, y, maxh):
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


def drawTower(tracedLine, i, S, xc, yc):
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


def drawVillage(tracedLine, i, S, xc, yc):
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




#Draws all doodles onto given line

def drawLandscape():
    global isDrawing
    isDrawing = True

    #gets the set of points that the axidraw will consider to be the hand drawn line
    tracedLine = findLine()
    if tracedLine == None:
        print("failed to find line")
        axi.moveto(0, 0)
        return

    #bugfixing
    axi.moveto(0, 0)
    return
    
    #AXIDRAW
    axi.moveto(0, 0)

    d = 0
    xMin = tracedLine[0][0]
    xMax = tracedLine[-1][0]
    for i in range(len(tracedLine)):
        if i > d:
            alignTest = False
            [x, y] = tracedLine[i]
            if rand.random() < 0.1:
                print("Attempting: Striation")
                drawStriation(tracedLine, i, x, y, 4, 60, S, xc, yc)

            birdScale = 0.7
            if rand.random() < 0.01 and y > 2*birdScale:
                print("Drawing: Bird")
                drawBird(x, rand.uniform(10, (y-3*birdScale)*S+yc), birdScale, xc, yc, S)

            #generate flocks    
            if rand.random() < 0.003 and y > 2+3*birdScale:
                print("Drawing: Flock")
                stepSize = 60*S
                birdx = x
                birdy = rand.uniform(10, (y-3*birdScale)*S+yc)
                for i in range(rand.randrange(5, 30)):
                    birdx += rand.uniform(-1*stepSize, stepSize)
                    birdy += rand.uniform(-1*stepSize*S, stepSize*S)
                    drawBird(birdx, birdy, birdScale, xc, yc, S)


            featureGen = rand.random()
            if featureGen < 0.06:  # 6%
                print("Drawing: Tree")
                alignTest = True
                d = drawTree(i, x*S + xc, y*S + yc, -30*S)
            elif featureGen < 0.1: # 4%
                print("Drawing: Tower")
                alignTest = True
                d = drawTower(tracedLine, i, S, xc, yc)
            elif featureGen < 0.13: # 3%
                print("Attempting: Village")
                alignTest = True
                d = drawVillage(tracedLine, i, S, xc, yc)
            elif featureGen < 0.16: # 3%
                print("Attempting: Lake")
                d = drawLake(tracedLine, i, x, y, 8, 70, 4*S, S, xc, yc)

            # if alignTest:
            #     roi = frame[x-25:x+25, y-25:y+25]
            #     cv.imshow('Working Area', roi)
            


    axi.moveto(0, 0)
    isDrawing = False


# Test to see if new surface detection is working
def traceSurfaces(islandList):
    global isRunning
    
    for island in islandList:
        for surface in island.surfaces:
            [startX, startY] = surface[0]
            axi.penup()
            axi.moveto(startX*S+xc, startY*S+yc)
            axi.pendown()

            for i in range(0, len(surface), 4):
                if not isRunning:
                    print("Drawing interrupted!")
                    axi.moveto(0, 0)
                    return
                
                [x, y] = surface[i]
                axi.lineto(x*S+xc, y*S+yc)

            axi.penup()

    axi.moveto(0, 0)


# Begins the collaboration, gets island information from the raw image
def runCollaboration(frame):
    global isRunning
    # Preprocess image to optimize it for image processing
    src_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # Convert to grayscale
    src_gray = cv.blur(src_gray, (3, 3)) # Apply blur to smooth the edges and reduce noise

    # Runs imageConverter to get data about islands and surfaces in the drawn image
    islandList = imgc.main(src_gray, thresh)
    print(len(islandList))
    # Test attempting to trace the location of all island surfaces
    if useAxi:
        traceSurfaces(islandList)

    isRunning = False



# Opens thread for the collaboration
def beginCollaboration(frame):
    #Signals that the collaboration is in progress
    global isRunning
    print(isRunning)
    if not isRunning:
        isRunning = True
        threading.Thread(target=runCollaboration, args=(frame,), daemon=True).start()
    

# Closes the thread for the collaboration
def stopCollaboration():
    global isRunning
    isRunning = False



# Define the initial variables
xDef = 4656  #resolution of the camera
yDef = 3496

thresh = 180  #higher means more land

#controls the size of the cropped in image
cropXmin = 1965
cropYmin = 1476
cropW = 1007
cropH = 684

S = 300
xc = 0
yc = 0
isDrawing = False  #True when the Axidraw is running
isRunning = False  #True when runCollaboration thread is running

useVid = True  # Choose whether to use camera or internal file for capture
useAxi = False  # For bugfixing while away from axidraw, program only works correctly with val is True


# Initialize windows to display the results
cv.namedWindow('Live Video Feed', cv.WINDOW_AUTOSIZE)
cv.resizeWindow('Live Video Feed', 1920, 1080)
cv.createTrackbar('Threshold', 'Live Video Feed', thresh, 500, on_thresh)
cv.createTrackbar('xc', 'Live Video Feed', xc, 4000, on_xstrackbar)
cv.createTrackbar('yc', 'Live Video Feed', yc, 2000, on_ystrackbar)
cv.createTrackbar('S', 'Live Video Feed', S, 1000, on_Strackbar)

cv.namedWindow('Contours',  cv.WINDOW_AUTOSIZE)
cv.resizeWindow('Contours', 1920, 1080)
cv.createTrackbar('MinX', 'Contours', cropXmin, xDef, on_xMin)
cv.createTrackbar('MinY', 'Contours', cropYmin, yDef, on_yMin)
cv.createTrackbar('Width', 'Contours', cropW, xDef, on_cropW)
cv.createTrackbar('Height', 'Contours', cropH, yDef, on_cropH)


# CAMERA CALLIBRATION CODE
# with np.load('calibration_data.npz') as data:  #the camera correction as calculated with openCV in cameraCallibration.py
#     cameraMatrix = data['cameraMatrix']
#     distCoeffs = data['distCoeffs']

# print(cameraMatrix)
# print(distCoeffs)


# Define a video capture object
if useVid:
    vid = cv.VideoCapture(1, cv.CAP_DSHOW)
    vid.set(cv.CAP_PROP_FRAME_WIDTH, xDef)
    vid.set(cv.CAP_PROP_FRAME_HEIGHT, yDef)
    
    if not vid.isOpened():
        raise IOError("Cannot open webcam")
    
    vid.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)
    vid.set(cv.CAP_PROP_EXPOSURE, -7)


# Setup axidraw
if useAxi:
    axi = axidraw.AxiDraw()          # Initialize class
    axi.interactive()                # Enter interactive context
    if not axi.connect():            # Open serial port to AxiDraw;
        print("not connected")
        quit()
    print("connected!")
    axi.options.units = 2


# Gets the image file if not using camera feed
if not useVid:
    parser = argparse.ArgumentParser(description='Code for Finding contours in your image tutorial.')
    parser.add_argument('--input', help='', default='./sampleImages/16mpzoomin.jpg')
    args = parser.parse_args()
    
    src = cv.imread(cv.samples.findFile(args.input))
    if src is None:
        print('Could not open or find the image:', args.input)
        exit(0)



#execution loop

while True:
    # Get the frame that will be used to create the collaboration
    if useVid:
        ret, frame = vid.read()
        if not ret:
            print("Failed to grab frame")
            break
    else:
        frame = src

    # Show the live video feed
    #cv.imshow('Live Video Feed', frame)
    frame = cv.resize(frame[cropYmin:cropYmin+cropH, cropXmin:cropXmin+cropW], (cropW, cropH))

    cv.imshow('Contours', frame)

    # Break the loop when 'esc' key is pressed
    k = cv.waitKey(1) & 0xFF
    if k == 32:  #spacebar
        beginCollaboration(frame)
    elif k == 27:  #esc
        stopCollaboration()
    elif k == 113:  #q
        axi.moveto(0, 0)
        break

# Release the VideoCapture object and close display windows
if useVid:
    vid.release()
cv.destroyAllWindows()