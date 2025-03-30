import math
import numpy as np
from PIL import Image, ImageDraw
import random as rand



#finds the lowest point in the overhang in range l, r
def overhangHeight(overhang, l, r):
    overhangSec = overhang[l:r]
    overhangSec = (y for [y, _] in overhangSec)
    lowestOverhang = max(overhangSec)
    return lowestOverhang





def drawBird(draw, x, y, birdScale, bend):
    draw.arc([x-birdScale, y-birdScale, x+birdScale, y+birdScale], start=270-bend, end=270+bend, fill = 0)
    offset = 2 * birdScale * np.cos(math.radians(90-bend))
    draw.arc([x+(offset-birdScale), y-birdScale, x+(offset+birdScale), y+birdScale], start=270-bend, end=270+bend, fill = 0)



def drawTree(draw, i, x, y, maxh):
    h = rand.uniform(3, maxh*0.7)  #height of trunk
    trunk = [x, y, x, y-h]
    draw.line(trunk, fill=0)

    r = rand.uniform(h*0.15, h*0.3)  #radius of crown
    draw.ellipse([x-r, (y-h)-r, x+r, (y-h)+r], outline=0)

    return (i+int(r)+1, [trunk, r])  #return new d val and all info about tree

    # if r > maxh/1.5 and rand.random() < 0.2:  #if crown is big add branches
    #     axi.moveto(x, (y+h)-rand.uniform(r/2.2, r/8))
    #     axi.lineto(x-r/6,  y+h-rand.uniform(-r/8, r/8))
    #     axi.moveto(x, (y+h)-rand.uniform(r/2.2, r/8))
    #     axi.lineto(x+r/6,  y+h-rand.uniform(-r/8, r/8))
    #     axi.moveto(x, y+h)
    #     axi.lineto(x, y+h+r/8)

    # elif r > maxh/1.5 and rand.random() < 0.2:
    #     arc(x, y+h, r/2, r/2, np.pi/2, 3*(np.pi)/2, True)
    #     axi.moveto(x, y+h)
    #     axi.lineto(x, y+h+r/8)



def drawStriation(draw, tracedLine, i, x, y, minStria, maxStria):
    isStria = True
    for c in range(minStria):
        if i+c >= len(tracedLine):
            isStria = False
            break

        if y < tracedLine[i+c][0]:
            isStria = False
    
    rX = -1
    if isStria:
        for c in range(minStria, maxStria):
            if i+c >= len(tracedLine):
                break

            if y < tracedLine[i+c][0]:
                rX = i + c
                break
    
    if rX != -1:
        endpoints = [x, y, tracedLine[rX][1], y]
        draw.line(endpoints, fill=0)
        return endpoints  #returns the drawn line
    
    return -1  #returns a flag showing that no line was drawn



def drawFish(draw, x, y, fishLen, fishDir):
    fishR = fishLen/2
    arcOffset = fishR/2

    upperBody = [x-fishR, y-fishR+arcOffset, x+fishR, y+fishR+arcOffset]
    lowerBody = [x-fishR, y-fishR-arcOffset, x+fishR, y+fishR-arcOffset]
    
    draw.arc(upperBody, 210, 330, fill = 0)
    draw.arc(lowerBody, 30, 150, fill = 0)

    upperBody = upperBody + [210, 330]
    lowerBody = lowerBody + [30, 150]
    
    tailStart = fishR*0.86602540378  #fishR*(sqrt(3)/2)
    tailEnd = tailStart + fishR*0.288675134595  #tailStart + fishR/(sqrt(3)*2)

    if fishDir == 1:
        tail = [x - tailStart, y, x - tailEnd, y - arcOffset, x - tailEnd, y + arcOffset]
    elif fishDir == -1:
        tail = [x + tailStart, y, x + tailEnd, y - arcOffset, x + tailEnd, y + arcOffset]
    else:
        print("Error: Invalid fish direction")


    draw.polygon(tail)

    return (upperBody, lowerBody, tail)


def generateAllFish(draw, allDepths, fishProb, fishSize, fishDir, startX, startY):
    buffer = 1 #how far away from walls and surface should fish be
    stepSize = fishSize + buffer + 2  #for the randomwalk school of fish, how far to walk between each fish
    lakeW = len(allDepths)  #how wide the lake is
    fishR = int(fishSize/2)

    edgeBuffer = fishR + buffer  #how close to the boundaries of the lake a fish can be
    
    if lakeW <= 2*edgeBuffer:  #exit if lake is too small for fish
        return
    
    i = edgeBuffer
    while i in range(edgeBuffer, lakeW - edgeBuffer):
        if allDepths[i] > 2*buffer + fishR and rand.random() < fishProb:
            xIdx = i
            y = startY + allDepths[i]/2

            minIdx = edgeBuffer
            maxIdx = lakeW - edgeBuffer
            minY = startY + edgeBuffer

            maxCurIdx = i
            while(minIdx <= xIdx < maxIdx and minY <= y < startY + (allDepths[xIdx] - edgeBuffer)):
                drawFish(draw, xIdx+startX, y, fishSize, fishDir)
                xStep = rand.randint(-stepSize, stepSize)
                xIdx += xStep
                yStepR = stepSize-xStep
                y += rand.choice([-yStepR, yStepR])

                maxCurIdx = max(maxCurIdx, xIdx)

            i = maxCurIdx
        i += 1



def drawLakeFeatures(draw, lineArr, startIdx, x, y, waveLen, endIdx):
    lakeW = (endIdx - startIdx)
    waveNum = int(lakeW/waveLen)
    leftOver = lakeW%waveLen
    waveD = waveLen + leftOver/waveNum

    waveInfo = [x, lineArr[endIdx][1], y, waveD, waveNum]  #start x, end x, y, wave width, number of waves
    boatInfo = -1

    draw.line([x, y, x+waveD, y])  #maybe remove?

    boatloc = int(rand.randrange(0, int((waveNum-2)*2)))
    for i in range(waveNum-2):

        #draw a wave
        xminBound = x+waveD*(i+1)
        draw.arc([xminBound, y-waveD/2, xminBound+waveD, y+waveD/2], 0, 180, fill = 0)
                
        if i == boatloc:  #generate boats
            boatScale = (3*waveLen + waveNum/4)/2  #half the total length of the hull of the boat
            boatX = x+waveD*i+waveD*1.5
            boatY = y - boatScale/4
            sailDir = rand.random() < 0.5
            
            #draw hull
            draw.arc([boatX-boatScale, boatY-boatScale/2, boatX+boatScale, boatY+boatScale/2], 0, 180, fill = 0)
            draw.line([boatX-boatScale, boatY, boatX+boatScale, boatY])

            #draw sail
            if sailDir:
                draw.polygon([boatX-boatScale/3, boatY, boatX-boatScale/3, boatY-(2*boatScale), boatX+(2/3)*boatScale, boatY])
            else:
                draw.polygon([boatX+boatScale/3, boatY, boatX+boatScale/3, boatY-(2*boatScale), boatX-(2/3)*boatScale, boatY])

            boatInfo = [boatX, boatScale, sailDir]

    draw.line([lineArr[endIdx][1]-waveD, y, lineArr[endIdx][1], y])  #maybe remove?

    return ([waveInfo, boatInfo])



def drawLake(draw, tracedLine, i, x, y, minLake, maxLake, waveLen):
    minDepth = 5  #at minimum how deep does a divot have to be generate a lake

    allDepths = []  #a list of the depth at every pixel of the lake
    isLake = True
    maxDetectedDepth = 0  #how deep this divot is
    for c in range(minLake):
        if i+c >= len(tracedLine):
            isLake = False
            break

        curY = tracedLine[i+c][0]
        
        if y > curY:
            isLake = False

        depth = curY-y
        allDepths.append(depth)
        maxDetectedDepth = max(maxDetectedDepth, curY-y)
    
    endIdx = 0
    if isLake:
        for c in range(minLake, maxLake):
            if i+c >= len(tracedLine):
                break

            curY = tracedLine[i+c][0]

            if y > curY:
                endIdx = i + c
                break
            
            depth = curY-y
            allDepths.append(depth)
            maxDetectedDepth = max(maxDetectedDepth, curY-y)
    

    if endIdx != 0 and maxDetectedDepth >= minDepth:  #create a lake if in bounds and deep enough
        lakeInfo = drawLakeFeatures(draw, tracedLine, i, x, y, waveLen, endIdx)

        generateAllFish(draw, allDepths, 0.05, 10, rand.choice([-1, 1]), x, y)

        return (endIdx, lakeInfo)

    return (endIdx, -1)



def drawTower(draw, tracedLine, overhang, i):
    minHeight = 8  #the minimum amount of space below island overhang to allow for a tower
    maxWidth = 8  #half the max width of a tower
    minWallH, maxWallH = 7, 60  #Absolute max height of tower walls
    wallHPercent = 0.7  #How much of availableH the walls of a tower can use

    #find left and right wall of tower
    l = i - rand.randrange(1, maxWidth)
    r = i + rand.randrange(1, maxWidth)

    #make sure tower sides are in bounds of tracedLine
    l = max(0, l)
    r = min(len(tracedLine) - 1, r)

    #find lowest overhang from another island
    lowestOverhang = overhangHeight(overhang, l, r)

    #determines dimensions of tower
    lx = tracedLine[l][1]
    ly = tracedLine[l][0]
    rx = tracedLine[r][1]
    ry = tracedLine[r][0]

    availableH = min(ly, ry) - lowestOverhang  #Space between floor and roof
    if availableH <= minHeight:  #Cancel if this is too small
        return (i, -1)
    
    h = rand.uniform(minWallH, maxWallH)  #gets unconstrained wall height
    h = min(h, availableH*wallHPercent)  #checks if needs to be shorter because of overhang
    h = ry - h  #converts to a y-coordinate

    leftWall = [lx, ly, lx, h]
    rightWall = [rx, ry, rx, h]

    draw.line(leftWall)
    draw.line(rightWall)

    w = rx-lx  #tower width
    remainingH = h-lowestOverhang-1  #remaining clearence below overhang
    if rand.random() < 0.3:  #square roof
        wh = w * rand.uniform(0.2, 1.2)  #roof overhang
        ht = w*rand.uniform(0.5, 1.5)  #roof height
        ht = min(ht, remainingH)  #makes sure roof doesn't go into island overhang
        
        roof = [lx-wh, h-ht, rx+wh, h]

        draw.rectangle(roof)

        d = i + w/2 + wh+1
        
    else:  #triangular roof
        wt = w * rand.uniform(1.2, 2.2)  #roof width
        ht = min((ry - h)/1.5, w*rand.uniform(0.7, 3))  #roof height
        ht = min(ht, remainingH)  #makes sure roof doesn't go into island overhang
        roof = [(lx+rx)/2 - wt/2, h, (lx+rx)/2 + wt/2, h, (lx+rx)/2, h - ht]

        draw.polygon(roof)
        
        d = i + int(wt/2)+1

    return (d, [leftWall, rightWall, roof])


def drawVillage(draw, tracedLine, overhang, i):
        minW, maxW = 5, 20
        minWallH, maxWallH = 7, 24
        minHeight = 8  #the minimum amount of space below island overhang to allow for a tower
        minRoofHeight = 2  #the minimum height of a roof (length from base to peak of roof)
        c = 0
        villageReceipt = []
        while i + c + maxW + 1 < len(tracedLine) and abs(tracedLine[i+c][0] - tracedLine[i+c+1][0]) < 2 and rand.random() > 0.06:

            w = rand.randint(minW, maxW)  #house walls width

            lowestOverhang = overhangHeight(overhang, i+c, i+c+w) + 1  #finds the lowest point of overhang
            hb = tracedLine[i+c][0] - rand.uniform(minWallH, maxWallH)  #house walls height
            hb = max(hb, lowestOverhang + minRoofHeight + 2)

            #coords of the base of left and right walls
            lx = tracedLine[i+c][1]
            ly = tracedLine[i+c][0]
            rx = tracedLine[i+c+w][1]
            ry = tracedLine[i+c+w][0]

            availableH = min(ly, ry) - lowestOverhang  #Space between floor and roof
            if availableH <= minHeight:  #Cancel if this is too small
                return (i+c+1, villageReceipt)

            leftWall = [lx, ly, lx, hb]
            rightWall = [rx, ry, rx, hb]

            #draw walls of house
            draw.line(leftWall)
            draw.line(rightWall)

            wt = w*rand.uniform(1, 1.4)  #roof width
            ht = wt*rand.uniform(0.4, 0.7)  #unconstrained roof height
            roofPeak = max(hb - ht, lowestOverhang)  #makes sure roof doesn't clip into ceiling overhang

            # draw house roof
            roof = [(lx+rx)/2 - (wt/2), hb, (lx+rx)/2 + (wt/2), hb, (lx+rx)/2, roofPeak]
            draw.polygon(roof)

            villageReceipt.append(("To", [leftWall, rightWall, roof]))
            c+=rand.randint(2, 5)  #start position of next house
        
        d = i+c+1
        return (d, villageReceipt)