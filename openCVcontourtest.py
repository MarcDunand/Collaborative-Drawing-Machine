from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng
import statistics as stat
 
rng.seed(12345)




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

    if(len(groupedDots) != 4):
        print("ERROR: length of orienting dot array is ", len(groupedDots))
        return
    
    dotF = [coordMean(i) for i in groupedDots]

    for i in range(4):
        dotF[i][0] = int(dotF[i][0])
        dotF[i][1] = int(dotF[i][1])

    dot1 = dotF[0]

    minxDiff = 100
    minyDiff = 100
    closeX = -1
    closeY = -1
    for i in range(1, 4):
        if abs(dot1[0]-dotF[i][0]) < minxDiff:
            minxDiff = abs(dot1[0]-dotF[i][0])
            closeX = i

        if abs(dot1[1]-dotF[i][1]) < minyDiff:
            minyDiff = abs(dot1[1]-dotF[i][1])
            closeY = i

    matchIdx = np.setdiff1d([1, 2, 3], [closeX, closeY])[0]
    match = dotF[matchIdx]

    dotF.pop(matchIdx)
    dotF.pop(0)


    pairs = [[dot1, match], dotF]

    pairs[0][0] = [pairs[0][0]]
    pairs[0][1] = [pairs[0][1]]
    pairs[1][0] = [pairs[1][0]]
    pairs[1][1] = [pairs[1][1]]

    corrected_pairs = [np.array(pair, dtype=np.int32) for pair in pairs]
    
    return corrected_pairs


def findIntersection(pairs):
    x1, y1 = pairs[0][0][0]
    x2, y2 = pairs[0][1][0]
    x3, y3 = pairs[1][0][0]
    x4, y4 = pairs[1][1][0]

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    num1 = (x1 * y2 - y1 * x2)
    num2 = (x3 * y4 - y3 * x4)
    numx = num1 * (x3 - x4) - (x1 - x2) * num2
    numy = num1 * (y3 - y4) - (y1 - y2) * num2

    ix = numx / den
    iy = numy / den

    return (int(ix), int(iy))


 
# Load source image
parser = argparse.ArgumentParser(description='Code for Finding contours in your image tutorial.')
parser.add_argument('--input', help='', default='IMG_1708.jpg')
args = parser.parse_args()
 
src = cv.imread(cv.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
 
# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3,3))
 
# Create Window
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)
max_thresh = 100
thresh = 70 # initial threshold

# Detect edges using Canny
canny_output = cv.Canny(src_gray, thresh, thresh * 2)

# Find contours
contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


# Draw contours
drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
for i in range(1, len(contours)):
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
    print(len(contours[i]))


for i in range(len(contours[1])):
    print(contours[1][i])



pairs = findDot(contours)

cv.drawContours(drawing, pairs, -1, (255, 255, 255), 1, cv.LINE_8)


intersection = findIntersection(pairs)

cv.circle(drawing, intersection, 6, (255, 255, 255), 1)



cv.imshow('Contours', drawing)


cv.waitKey()