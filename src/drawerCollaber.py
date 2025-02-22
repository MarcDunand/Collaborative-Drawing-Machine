import cv2 as cv
import numpy as np
import random as rand
import statistics as stat
import math
from pyaxidraw import axidraw  
from scipy.interpolate import splprep, splev
import keyboard
import argparse


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



#returns the contour that corresponds to the hand drawn line

def getHandDrawnLine(frame, thresh):

    # Convert to grayscale
    src_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    src_gray = cv.blur(src_gray, (3, 3))

    # Perform Canny edge detection
    canny_output = cv.Canny(src_gray, thresh, thresh * 2)

    # Find contours
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours_img = np.zeros_like(frame)

    # Drawing the contours on the empty image
    for i in range(len(contours)):
        color = (255, 255, 255)
        cv.drawContours(contours_img, contours, i, color, 0, cv.LINE_8, hierarchy, 0)

    #cv.imshow('ContoursTest', contours_img)


    maxSq = []
    maxWH = -1
    for cnt in contours:
        epsilon = 0.05 * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:  # Check for quadrilateral
            # Further checks can be area, aspect ratio
            (x, y, w, h) = cv.boundingRect(cnt)
            aspect_ratio = float(w)/h
            if 1 < aspect_ratio < 1.6:  # Roughly square
                if w*h > maxWH:
                    maxWH = w*h
                    maxSq = cnt

    if maxWH != -1:
        pgBd = cv.boundingRect(maxSq)
        possLines = []
        for cnt in contours:
            cntBd = cv.boundingRect(cnt)
            if pgBd[0] < cntBd[0] and pgBd[1] < cntBd[1] and pgBd[0] + pgBd[2] > cntBd[0] + cntBd[2] and pgBd[1] + pgBd[3] > cntBd[1] + cntBd[3]:
                possLines.append(cnt)
        if len(possLines) > 0:
            return max(possLines, key=len)
        else:
            return np.array([])
    else:
        return np.array([])
 

#turns the landscape contour into an interpretable set of points

def makeFunction(baseLoop):
    loopLine = []
    for elem in baseLoop:
        loopLine.append((elem[0]).tolist())


    xInc = loopLine[0] < loopLine[1]

    line1 = []
    line2 = []

    #parses the closed countour into the upper and lower edges of the line (line1 and line2)
    if xInc:
        idx = 0
        while loopLine[idx][0] <= loopLine[idx+1][0]:
            line1.append(loopLine[idx])
            idx += 1
        
        idx += 5

        while idx < len(loopLine) - 1 and loopLine[idx][0] >= loopLine[idx+1][0]:
            line2.append(loopLine[idx])
            idx += 1

        idx += 5

        while idx < len(loopLine):
            line1.append(loopLine[idx])
            idx += 1

    else:
        idx = 0
        while loopLine[idx][0] >= loopLine[idx+1][0]:
            line1.append(loopLine[idx])
            idx += 1
        
        idx += 5

        while idx < len(loopLine) - 1 and loopLine[idx][0] <= loopLine[idx+1][0]:
            line2.append(loopLine[idx])
            idx += 1
        idx += 5

        while idx < len(loopLine):
            line1.append(loopLine[idx])
            idx += 1


    if len(line1) < 2 or len(line2) < 2:
        print("Detected sublines are too short (is most likely detecting wrong contour)")
        return None
    
    line1.sort(key=lambda coord: coord[0])
    line2.sort(key=lambda coord: coord[0])


    line1 = list({x[0]: x for x in line1}.values())
    line2 = list({x[0]: x for x in line2}.values())

    # Ensure the list is sorted by x coordinate

    interp1 = [line1[0]]

    for i in range(1, len(line1)):
        current = line1[i]
        previous = line1[i-1]

        x_diff = current[0] - previous[0]

        if x_diff > 1:
            for j in range(1, x_diff):
                # Linear interpolation formula: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
                interpolated_y = previous[1] + (j * (current[1] - previous[1]) / x_diff)
                # Append the new interpolated coordinate
                interp1.append([previous[0] + j, interpolated_y])

        interp1.append(current)


    interp2 = [line2[0]]

    for i in range(1, len(line2)):
        current = line2[i]
        previous = line2[i-1]

        x_diff = current[0] - previous[0]

        if x_diff > 1:
            for j in range(1, x_diff):
                # Linear interpolation formula: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
                interpolated_y = previous[1] + (j * (current[1] - previous[1]) / x_diff)
                # Append the new interpolated coordinate
                interp2.append([previous[0] + j, interpolated_y])

        interp2.append(current)

    try:
        startDiff = interp1[0][0] - interp2[0][0]
        if startDiff < 0:
            interp1 = interp1[abs(startDiff):]
        else:
            interp2 = interp2[abs(startDiff):]

        endDiff = interp1[-1][0] - interp2[-1][0]
        if endDiff > 0:
            interp1 = interp1[0:len(interp1) - abs(endDiff)]
        else:
            interp2 = interp2[0:len(interp2) - abs(endDiff)]

    except:  #if there's a problem with finding the line just abort
        return None


    avgLine = []
    for i in range(len(interp1)):
        avgLine.append([interp1[i][0], (interp1[i][1] + interp2[i][1])/2])

    return avgLine


#gets the set of points that the axidraw will consider to be the hand drawn line
def findLine():
    if useVid:
        ret, frame = vid.read()
        if not ret:
            print("Failed to grab frame")
            return None

        scale = frame.shape[:2]
        newCameraMatrix, roi = cv. getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, scale, 1, scale)
        dst = cv.undistort(frame, cameraMatrix, distCoeffs, None, newCameraMatrix)
        x, y, w, h = roi
        #dst = dst[y:y+h, x:x+w]
        cv.imwrite('calibresult.png', dst)
        cv.imwrite('controlimg.png', frame)
        

        handLine = getHandDrawnLine(dst, thresh)
        
    else:
        handLine = getHandDrawnLine(src, thresh)
    if handLine.size == 0:
        print("ERROR: failed to find handrawn line")
        return None

    if handLine.size < 30:
        print("ERROR: handrawn line is small (less than 30 pts)")
    
    tracedLine = makeFunction(handLine)
    if tracedLine == None:
        print("ERROR: failed to makeFunction from detected line")
        return None

    #traces the handdrawn line (for bugfixing)
    [startX, startY] = tracedLine[0]
    axi.moveto(startX*S+xc, startY*S+yc)
    for elem in tracedLine:
        [x, y] = elem
        axi.lineto(x*S+xc, y*S+yc)
    axi.penup()

    return tracedLine





#current execution controls

def on_space(event):
    drawLandscape()

keyboard.on_press_key("space", on_space)

def on_trackbar(val):
    global thresh
    thresh = val

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



#Program setup
with np.load('calibration_data.npz') as data:  #the camera correction as calculated with openCV in cameraCallibration.py
    cameraMatrix = data['cameraMatrix']
    distCoeffs = data['distCoeffs']

print(cameraMatrix)
print(distCoeffs)

print("Awake!")

useVid = True

# Define a video capture object
if useVid:
    vid = cv.VideoCapture(0, cv.CAP_DSHOW)
    vid.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    vid.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

axi = axidraw.AxiDraw()          # Initialize class
axi.interactive()                # Enter interactive context
if not axi.connect():            # Open serial port to AxiDraw;
    print("not connected")
    quit()
print("connected!")
axi.options.units = 2


# Check if the webcam is opened correctly
if useVid and not vid.isOpened():
    raise IOError("Cannot open webcam")



# Define the initial variables
thresh = 100
prevSq = (0, 0)
S = 375  
xc = 2115
yc = 311
isDrawing = False

# with np.load('calibration_data.npz') as data:  #the camera correction as calculated with openCV in cameraCallibration.py
#     cameraMatrix = data['cameraMatrix']
#     distCoeffs = data['distCoeffs']

# distCoeffs[0][0] -= 0.01
# distCoeffs[0][1] -= 0.01


print(distCoeffs)


# Create windows to display the results
cv.namedWindow('Live Video Feed', cv.WINDOW_AUTOSIZE)
cv.resizeWindow('Live Video Feed', 1920, 1080)
cv.createTrackbar('xc', 'Live Video Feed', xc, 4000, on_xstrackbar)
cv.createTrackbar('yc', 'Live Video Feed', yc, 2000, on_ystrackbar)
cv.createTrackbar('S', 'Live Video Feed', S, 1000, on_Strackbar)

cv.namedWindow('Contours',  cv.WINDOW_AUTOSIZE)
cv.resizeWindow('Contours', 1920, 1080)
cv.createTrackbar('Value', 'Contours', thresh, 500, on_trackbar)



if not useVid:
    parser = argparse.ArgumentParser(description='Code for Finding contours in your image tutorial.')
    parser.add_argument('--input', help='', default='screenGrab.jpg')
    args = parser.parse_args()
    
    src = cv.imread(cv.samples.findFile(args.input))
    if src is None:
        print('Could not open or find the image:', args.input)
        exit(0)




#execution loop

while True:
    if useVid:
        ret, frame = vid.read()
        if not ret:
            print("Failed to grab frame")
            break
    else:
        frame = src

    yi = 650
    xi = 1050



    # Convert to grayscale
    src_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


    src_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Define the range of yellow color in HSV
    lower_red1 = (0, 100, 25)    # Lower bound of lower red range (H, S, V)
    upper_red1 = (10, 255, 255)  # Upper bound of lower red range (H, S, V)
    lower_red2 = (150, 100, 25)  # Lower bound of upper red range (H, S, V)
    upper_red2 = (179, 255, 255) # Upper bound of upper red range (H, S, V)

    # Threshold the HSV image to get only red colors
    mask1 = cv.inRange(src_hsv, lower_red1, upper_red1)
    mask2 = cv.inRange(src_hsv, lower_red2, upper_red2)
    thresholded = cv.bitwise_or(mask1, mask2)

    contours, _ = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Apply blur to smooth the edges and reduce noise
    src_gray = cv.blur(src_gray, (3, 3))

    # Perform Canny edge detection
    canny_output = cv.Canny(src_gray, thresh, thresh * 2)

    # Find contours
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Create an empty image for drawing contours
    contours_img = np.zeros_like(frame)

    # Drawing the contours on the empty image
    for i in range(len(contours)):
        color = (255, 255, 255)
        cv.drawContours(contours_img, contours, i, color, 0, cv.LINE_8, hierarchy, 0)

    # Show the contours in a separate window
    cv.imshow('Contours', contours_img)      


    
    #shows the line that the axidraw considers to be the location of the hand drawn line
    showPreviewLine = False
    if showPreviewLine and not isDrawing:
        # Determine the points that the axidraw would aim for when activated
        tracedLine = findLine()
        
        #adjust size and position of tracedLine to be accurate to how the axidraw moves



        if tracedLine != None:
            # Reshape the points array to the required shape for cv.polylines
            tracedLine = np.array(tracedLine, np.int32).reshape((-1, 1, 2))

            # Draw the polyline on the frame
            # (frame, [points], isClosed, color, thickness)
            cv.polylines(frame, [tracedLine], isClosed=False, color=(0, 0, 255), thickness=1)



    # Show the live video feed
    cv.imshow('Live Video Feed', frame)  


    # Break the loop when 'esc' key is pressed
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

# Release the VideoCapture object and close display windows
if useVid:
    vid.release()
cv.destroyAllWindows()