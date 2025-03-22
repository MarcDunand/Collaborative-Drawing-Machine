import math
import numpy as np
from PIL import Image, ImageDraw
import random as rand


def drawBird(draw, x, y, birdScale, bend):
    draw.arc([x-birdScale, y-birdScale, x+birdScale, y+birdScale], start=270-bend, end=270+bend, fill = 0)
    offset = 2 * birdScale * np.cos(math.radians(90-bend))
    draw.arc([x+(offset-birdScale), y-birdScale, x+(offset+birdScale), y+birdScale], start=270-bend, end=270+bend, fill = 0)



def drawTree(draw, i, x, y, maxh):
    h = rand.uniform(3, maxh)  #height of trunk
    draw.line([x, y, x, y-h], fill=0)
    r = rand.uniform(h/7, h/2)  #radius of crown
    #circle(x, y + h, r)  #draw crown
    draw.ellipse([x-r, (y-h)-r, x+r, (y-h)+r], outline=0)
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

    return i+int(r)+1



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
        print("Drawing: Striation")
        draw.line([x, y, tracedLine[rX][1], y], fill=0)




def drawBoats(draw, lineArr, startIdx, x, y, waveLen, endIdx):
    lakeW = (endIdx - startIdx)
    waveNum = int(lakeW/waveLen)
    leftOver = lakeW%waveLen
    waveD = waveLen + leftOver/waveNum

    draw.line([x, y, x+waveD, y])  #maybe remove?

    boatloc = int(rand.randrange(0, int((waveNum-2)*2)))
    for i in range(waveNum-2):

        #draw one wave
        xminBound = x+waveD*(i+1)
        draw.arc([xminBound, y-waveD/2, xminBound+waveD, y+waveD/2], 0, 180, fill = 0)
                
        if i == boatloc:  #generate boats
            print("Drawing: Boat")
            boatScale = (3*waveLen + waveNum/7)/2  #half the total length of the hull of the boat
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

    draw.line([lineArr[endIdx][1]-waveD, y, lineArr[endIdx][1], y])  #maybe remove?



def drawLake(draw, tracedLine, i, x, y, minLake, maxLake, waveLen):
    isLake = True
    for c in range(minLake):
        if i+c >= len(tracedLine):
            isLake = False
            break

        if y > tracedLine[i+c][0]:
            isLake = False
    
    endIdx = 0
    if isLake:
        for c in range(minLake, maxLake):
            if i+c >= len(tracedLine):
                break

            if y > tracedLine[i+c][0]:
                endIdx = i + c
                break
    
    if endIdx != 0:
        print("Drawing: Lake")
        drawBoats(draw, tracedLine, i, x, y, waveLen, endIdx)

    return endIdx



def drawTower(draw, tracedLine, i):
    #find left and right wall of tower
    l = i - rand.randrange(1, 4)
    r = i + rand.randrange(1, 4)

    #make sure tower sides are in bounds of tracedLine
    l = max(0, l)
    r = min(len(tracedLine) - 1, r)

    #determines dimensions of tower
    lx = tracedLine[l][1]
    ly = tracedLine[l][0]
    rx = tracedLine[r][1]
    ry = tracedLine[r][0]
    h = (tracedLine[r][0]) - rand.uniform(6, 30)
    w = rx-lx

    draw.line([lx, ly, lx, h])
    draw.line([rx, ry, rx, h])

    
    if rand.random() < 0.3:  #square roof
        wh = w * rand.uniform(0.2, 1.2)  #roof overhang
        ht = wh*rand.uniform(0.5, 1.5)  #roof height
        draw.rectangle([lx-wh, h-ht, rx+wh, h], fill = None)
        d = rx+wh+1
        
    else:  #triangular roof
        wt = w * rand.uniform(1.2, 2.2)  #roof width
        ht = min((ry - h)/1.5, w*rand.uniform(0.7, 3))  #roof height
        draw.polygon([(lx+rx)/2 - wt/2, h, (lx+rx)/2 + wt/2, h, (lx+rx)/2, h - ht])
        d = i + int(wt/2)+1

    return d


def drawVillage(draw, tracedLine, i):
        c = 0
        while abs(tracedLine[i+c][0] - tracedLine[i+c+1][0]) < 2:
            print("Drawing: Village House")
            w = rand.randint(3, 10)  #house walls width
            if i + c + 8 >= len(tracedLine):  #end village if too close to end of line
                break

            hb = tracedLine[i+c][0] - rand.uniform(4, 12)  #house walls height

            #coords of the base of left and right walls
            lx = tracedLine[i+c][1]
            ly = tracedLine[i+c][0]
            rx = tracedLine[i+c+w][1]
            ry = tracedLine[i+c+w][0]

            #draw walls of house
            draw.line([lx, ly, lx, hb])
            draw.line([rx, ry, rx, hb])

            wt = w*rand.uniform(1, 1.4)
            ht = wt*rand.uniform(0.4, 0.7)

            # draw house roof
            draw.polygon([(lx+rx)/2 - (wt/2), hb, (lx+rx)/2 + (wt/2), hb, (lx+rx)/2, hb - ht])

            c+=rand.randint(2, 5)  #start position of next house
        
        d = i+c+1
        return d