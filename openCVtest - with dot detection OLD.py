import cv2 as cv
import numpy as np
import random as rand
import statistics as stat
import math
from pyaxidraw import axidraw  
from scipy.interpolate import splprep, splev
import keyboard


print("Awake!")

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


def coordMean(coords):
    xs = []
    ys = []
    for i in range(len(coords)):
        xs.append(coords[i][0])
        ys.append(coords[i][1])
    return [stat.mean(map(float, xs)), stat.mean(map(float, ys))]

def findDot(contours):
    dots = []
    for i in range(len(contours)):
        contour = contours[i]
        contour = [i[0] for i in contour]
        if len(contour) < 30:
            dots.append(coordMean(contour))
    
    
    groupedDots = []
    for i in range(len(dots)):
        dot = dots[i]
        foundMatch = False
        for j in range(len(groupedDots)):
            if abs(dot[0] - groupedDots[j][0][0]) < 5 and abs(dot[1] - groupedDots[j][0][1]) < 5:
                groupedDots[j].append(dot)
                foundMatch = True
                break

        if not foundMatch:
            groupedDots.append([dot])
    
    dotF = [coordMean(i) for i in groupedDots]


    for i in range(len(dotF)):
        dotF[i][0] = int(dotF[i][0])
        dotF[i][1] = int(dotF[i][1])


    if(len(dotF) > 0):
        bins = [[dotF[0]]]
        
        for i in range(1, len(dotF)):
            foundBin = False
            for j in range(len(bins)):
                dist = math.dist(dotF[i], bins[j][0])
                if 5 < dist < 30:
                    bins[j].append(dotF[i])
                    foundBin = True

            if not foundBin:
                bins.append([dotF[i]])
            
        squIdx = -1
        for i in range(len(bins)):
            if len(bins[i]) == 1:
                squIdx = i

        if squIdx == -1:
            print("ERROR: no dot square found", bins)
            return None
        else:
            return bins[squIdx][0]
        
        # dotF = bins[squIdx]

        # dot1 = dotF[0]

        # minxDiff = 100
        # minyDiff = 100
        # closeX = -1
        # closeY = -1
        # for i in range(1, 4):
        #     if abs(dot1[0]-dotF[i][0]) < minxDiff:
        #         minxDiff = abs(dot1[0]-dotF[i][0])
        #         closeX = i

        #     if abs(dot1[1]-dotF[i][1]) < minyDiff:
        #         minyDiff = abs(dot1[1]-dotF[i][1])
        #         closeY = i

        # matchIdx = np.setdiff1d([1, 2, 3], [closeX, closeY])[0]
        # match = dotF[matchIdx]

        # dotF.pop(matchIdx)
        # dotF.pop(0)


        # pairs = [[dot1, match], dotF]

        # pairs[0][0] = [pairs[0][0]]
        # pairs[0][1] = [pairs[0][1]]
        # pairs[1][0] = [pairs[1][0]]
        # pairs[1][1] = [pairs[1][1]]

        # corrected_pairs = [np.array(pair, dtype=np.int32) for pair in pairs]
        
        # return corrected_pairs
    else:
        return None


def findIntersection(pairs):
    x1, y1 = pairs[0][0][0]
    x2, y2 = pairs[0][1][0]
    x3, y3 = pairs[1][0][0]
    x4, y4 = pairs[1][1][0]

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if den == 0:
        print("ERROR: div by 0")
        return (x1, y1)

    num1 = (x1 * y2 - y1 * x2)
    num2 = (x3 * y4 - y3 * x4)
    numx = num1 * (x3 - x4) - (x1 - x2) * num2
    numy = num1 * (y3 - y4) - (y1 - y2) * num2

    ix = numx / den
    iy = numy / den

    return (int(ix), int(iy))


def getHandDrawnLine(vid, thresh):
    ret, frame = vid.read()
    if not ret:
        print("Failed to grab frame")
        return np.array([])

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

    cv.imshow('ContoursTest', contours_img)


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
        for cnt in contours:
            cntBd = cv.boundingRect(cnt)
            if pgBd[0] < cntBd[0] and pgBd[1] < cntBd[1] and pgBd[0] + pgBd[2] > cntBd[0] + cntBd[2] and pgBd[1] + pgBd[3] > cntBd[1] + cntBd[3]:
                print(cnt)
                return cnt
        return np.array([])
    else:
        return np.array([])
 

def makeFunction(baseLoop):
    loopLine = []
    for elem in baseLoop:
        loopLine.append((elem[0]).tolist())

    print(loopLine)

    xInc = loopLine[0] < loopLine[1]

    line1 = []
    line2 = []
    if xInc:
        print("inc")
        idx = 0
        while loopLine[idx][0] <= loopLine[idx+1][0]:
            line1.append(loopLine[idx])
            idx += 1
        
        idx += 5

        while loopLine[idx][0] >= loopLine[idx+1][0]:
            line2.append(loopLine[idx])
            idx += 1

        idx += 5

        while idx < len(loopLine):
            line1.append(loopLine[idx])
            idx += 1

    else:
        print("not inc")
        idx = 0
        while loopLine[idx][0] >= loopLine[idx+1][0]:
            line1.append(loopLine[idx])
            idx += 1
        
        print("id: ", idx, loopLine[idx])
        idx += 5

        while loopLine[idx][0] <= loopLine[idx+1][0]:
            line2.append(loopLine[idx])
            idx += 1
        print("id: ", idx, loopLine[idx])
        idx += 5

        while idx < len(loopLine):
            line1.append(loopLine[idx])
            idx += 1

    

    line1.sort(key=lambda coord: coord[0])
    line2.sort(key=lambda coord: coord[0])

    print(line1, "llllllllllllllllll", line2)

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

    print(interp1, "\n\nAAAAAAAAAAA\n\n", interp2)

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

    print(interp1, "\n\nCCCCCCCCCCCCCCCCCC\n\n", interp2)

    print(len(interp1), len(interp2))

    avgLine = []
    for i in range(len(interp1)):
        avgLine.append([interp1[i][0], (interp1[i][1] + interp2[i][1])/2])

    return avgLine

        


    

# def camToAxi(x, y)



# Initialize the RNG with a fixed seed
#rand.seed(12345)
print("1")

# Define a video capture object
vid = cv.VideoCapture(1)

print("2")

axi = axidraw.AxiDraw()          # Initialize class
axi.interactive()                # Enter interactive context
if not axi.connect():            # Open serial port to AxiDraw;
    print("not connected")
    quit()
print("connected!")
axi.options.units = 2


# Check if the webcam is opened correctly
if not vid.isOpened():
    raise IOError("Cannot open webcam")

# Create windows to display the results
cv.namedWindow('Live Video Feed', cv.WINDOW_AUTOSIZE)
cv.namedWindow('Contours', cv.WINDOW_AUTOSIZE)

# Define the initial threshold for Canny edge detection
thresh = 100
prevSq = (0, 0)

def on_space(event):
    drawLandscape()

def drawTree(x, y, maxh):
    h = rand.uniform(3, maxh)  #height of trunk
    axi.moveto(x, y)
    axi.lineto(x, y+h)
    r = rand.uniform(h/7, h/1.5)  #radius of crown
    circle(x, y + h, r)  #draw crown
    if r > maxh/2.5 and rand.random() < 0.5:  #if crown is big add branches
        axi.moveto(x, (y+h)-rand.uniform(r/2.2, r/8))
        axi.lineto(x-r/6,  y+h-rand.uniform(-r/8, r/8))
        axi.moveto(x, (y+h)-rand.uniform(r/2.2, r/8))
        axi.lineto(x+r/6,  y+h-rand.uniform(-r/8, r/8))
        axi.moveto(x, y+h)
        axi.lineto(x, y+h+r/8)

    elif r > maxh/2.5:
        arc(x, y+h, r/2, r/2, np.pi/2, 3*(np.pi)/2)
        axi.moveto(x, y+h)
        axi.lineto(x, y+h+r/8)

    axi.penup()


def drawLandscape():
    handLine = getHandDrawnLine(vid, thresh)
    if handLine.size == 0:
        print("ERROR: failed to find handrawn line")
        return

    if handLine.size < 30:
        print("ERROR: handrawn line is small ()")
    
    tracedLine = makeFunction(handLine)


    #AXIDRAW
    axi.moveto(0, 0)

    S = 0.68
    xc = -46
    yc = -45

    axi.moveto(handLine[0][0][0]*S + xc  , handLine[0][0][1]*S + yc)
    for pt in handLine:
        axi.lineto(pt[0][0]*S + xc, pt[0][1]*S + yc)


    # for pt in tracedLine:
    #     if rand.random() < 0.08:
            #drawTree(pt[0]*S + xc, pt[1]*S + yc, -30*S)

            # axi.moveto(pt[0]*S + xc, pt[1]*S + yc)
            # axi.lineto(pt[0]*S + xc, (pt[1] - 15)*S + yc)
            # axi.penup()



    axi.moveto(0, 0)

keyboard.on_press_key("space", on_space)


    

while True:
    ret, frame = vid.read()
    if not ret:
        print("Failed to grab frame")
        break

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

    
    squareContours = []
    maxWHIdx = -1
    maxWH = -1
    contours, _ = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        epsilon = 0.05 * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:  # Check for quadrilateral
            # Further checks can be area, aspect ratio
            (x, y, w, h) = cv.boundingRect(cnt)
            aspect_ratio = float(w)/h
            if 0.8 < aspect_ratio < 1.3:  # Roughly square
                squareContours.append((x, y))

                if w*h > maxWH:
                    maxWH = w*h
                    maxWHIdx = len(squareContours) - 1

    # mask = np.zeros_like(src_gray)
    # cv.drawContours(mask, squareContours, -1, (255), thickness=cv.FILLED)
    # masked_image = cv.bitwise_and(src_gray, src_gray, mask=mask)


    # cv.imshow('thresh', masked_image)







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

    if len(squareContours) > 0:  
        newSq = (squareContours[maxWHIdx][0], squareContours[maxWHIdx][1])
        if newSq[0] != 0 and newSq[1] != 0:
            cv.circle(contours_img, newSq, 6, (255, 255, 255), 1)
            prevSq = newSq
        else:
            cv.circle(contours_img, prevSq, 6, (255, 255, 255), 1)
    else:
        cv.circle(contours_img, prevSq, 6, (255, 255, 255), 1)

    # Show the live video feed
    cv.imshow('Live Video Feed', frame)
    
    # Show the contours in a separate window
    cv.imshow('Contours', contours_img)        
    


    # paperMask = np.zeros_like(contours_img)
    # cv.drawContours(paperMask, [maxSq], -1, (255, 255, 255), thickness=cv.FILLED)
    # masked_image = cv.bitwise_and(contours_img, paperMask)
    # cv.imshow('PAGE', masked_image)


    # Break the loop when 'esc' key is pressed
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

# Release the VideoCapture object and close display windows
vid.release()
cv.destroyAllWindows()