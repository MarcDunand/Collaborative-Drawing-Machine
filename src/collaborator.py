import cv2 as cv
import numpy as np
import random as rand
import math
from pyaxidraw import axidraw
from scipy.interpolate import splprep, splev
from scipy.ndimage import convolve
import argparse
import imageConverter as imgc
from imageConverter import Island
import axiDoodles as do
import threading



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

useVid = False  # Choose whether to use camera or internal file for capture
useAxi = False  # For bugfixing while away from axidraw, program only works correctly with val is True



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






#Image processing techniques to convert with accuracy to b&w

#Fills in any pixel that is surrounded by 3 or more filled pixels
def fill_enclosed_pixels(binary_array):
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])  # 4-connected neighborhood
    changed = True
    while changed:
        neighbor_count = convolve(binary_array, kernel, mode='constant', cval=0)
        new_array = np.where((binary_array == 0) & (neighbor_count >= 3), 1, binary_array)
        changed = not np.array_equal(new_array, binary_array)
        binary_array = new_array
    return binary_array


#morphological dialation/erosion
def fill_holes_morph(binary_image, cell_size):
    kernel = np.ones((cell_size,cell_size), np.uint8)  # Small structuring element
    closed = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, kernel)
    return closed


def preprocess_image(frame, thresh):
    #first convert to greyscale
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #converts the image to b&w depending on the darkenss of the pixel and thresh
    binary_array = (gray_frame < thresh).astype(np.uint8)

    # Fill in any pixel that is surrounded by 3 or more filled pixels
    filled_holes_img = fill_enclosed_pixels(binary_array)

    # Use a morphological dialation/erosion system to close holes
    morpho_img = fill_holes_morph(filled_holes_img, 3)
    
    return morpho_img



#Draws all doodles onto given line
def drawLandscape(tracedLine):
    global isDrawing
    isDrawing = True

    #gets the set of points that the axidraw will consider to be the hand drawn line
    if tracedLine == None:
        print("failed to find line")
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
                do.drawStriation(axi, tracedLine, i, x, y, 4, 60, S, xc, yc)

            birdScale = 0.7
            if rand.random() < 0.01 and y > 2*birdScale:
                print("Drawing: Bird")
                do.drawBird(axi, x, rand.uniform(10, (y-3*birdScale)*S+yc), birdScale, xc, yc, S)

            #generate flocks    
            if rand.random() < 0.003 and y > 2+3*birdScale:
                print("Drawing: Flock")
                stepSize = 60*S
                birdx = x
                birdy = rand.uniform(10, (y-3*birdScale)*S+yc)
                for i in range(rand.randrange(5, 30)):
                    birdx += rand.uniform(-1*stepSize, stepSize)
                    birdy += rand.uniform(-1*stepSize*S, stepSize*S)
                    do.drawBird(axi, birdx, birdy, birdScale, xc, yc, S)


            featureGen = rand.random()
            if featureGen < 0.06:  # 6%
                print("Drawing: Tree")
                alignTest = True
                d = do.drawTree(axi, i, x*S + xc, y*S + yc, -30*S)
            elif featureGen < 0.1: # 4%
                print("Drawing: Tower")
                alignTest = True
                d = do.drawTower(axi, tracedLine, i, S, xc, yc)
            elif featureGen < 0.13: # 3%
                print("Attempting: Village")
                alignTest = True
                d = do.drawVillage(axi, tracedLine, i, S, xc, yc)
            elif featureGen < 0.16: # 3%
                print("Attempting: Lake")
                d = do.drawLake(axi, tracedLine, i, x, y, 8, 70, 4*S, S, xc, yc)

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

    # Runs imageConverter to get data about islands and surfaces in the drawn image
    islandList = imgc.main(frame)
    print(len(islandList))
    # Test attempting to trace the location of all island surfaces
    if useAxi:
        traceSurfaces(islandList)

    isRunning = False

    for island in islandList:
        for surface in island.surfaces:
            drawLandscape(surface)



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




# Initialize windows to display the results
cv.namedWindow('Contours and Correction', cv.WINDOW_AUTOSIZE)
cv.resizeWindow('Contours and Correction', 1920, 1080)
cv.createTrackbar('Threshold', 'Contours and Correction', thresh, 500, on_thresh)
cv.createTrackbar('xc', 'Contours and Correction', xc, 4000, on_xstrackbar)
cv.createTrackbar('yc', 'Contours and Correction', yc, 2000, on_ystrackbar)
cv.createTrackbar('S', 'Contours and Correction', S, 1000, on_Strackbar)

cv.namedWindow('Positioning',  cv.WINDOW_AUTOSIZE)
cv.resizeWindow('Positioning', 1920, 1080)
cv.createTrackbar('MinX', 'Positioning', cropXmin, xDef, on_xMin)
cv.createTrackbar('MinY', 'Positioning', cropYmin, yDef, on_yMin)
cv.createTrackbar('Width', 'Positioning', cropW, xDef, on_cropW)
cv.createTrackbar('Height', 'Positioning', cropH, yDef, on_cropH)


# CAMERA CALLIBRATION CODE
# with np.load('calibration_data.npz') as data:  #the camera correction as calculated with openCV in cameraCallibration.py
#     cameraMatrix = data['cameraMatrix']
#     distCoeffs = data['distCoeffs']

# print(cameraMatrix)
# print(distCoeffs)



if useVid:
    # Define a video capture object
    vid = cv.VideoCapture(1, cv.CAP_DSHOW)
    vid.set(cv.CAP_PROP_FRAME_WIDTH, xDef)
    vid.set(cv.CAP_PROP_FRAME_HEIGHT, yDef)
    
    if not vid.isOpened():
        raise IOError("Cannot open webcam")
    
    vid.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)
    vid.set(cv.CAP_PROP_EXPOSURE, -7)
else:
    # Gets the image file
    parser = argparse.ArgumentParser(description='Code for Finding contours in your image tutorial.')
    parser.add_argument('--input', help='', default='./sampleImages/16mpzoomin.jpg')
    args = parser.parse_args()
    
    src = cv.imread(cv.samples.findFile(args.input))
    if src is None:
        print('Could not open or find the image:', args.input)
        exit(0)


# Setup axidraw
if useAxi:
    axi = axidraw.AxiDraw()          # Initialize class
    axi.interactive()                # Enter interactive context
    if not axi.connect():            # Open serial port to AxiDraw;
        print("not connected")
        quit()
    print("connected!")
    axi.options.units = 2



#execution loop

while True:
    # Get the frame that will be used to create the collaboration
    if useVid:
        ret, frame = vid.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv.resize(frame[cropYmin:cropYmin+cropH, cropXmin:cropXmin+cropW], (cropW, cropH))
    else:
        frame = src
        
    #generates the frame that will be interpreted by the collaborator
    processed_frame = preprocess_image(frame, thresh)


    #shows live image
    cv.imshow('Positioning', frame)

    #shows the frame as it will be interpreted by the collaborator
    cv.imshow('Contours and Correction', np.where(processed_frame == 0, 255, 0).astype(np.uint8))


    # Key commands
    k = cv.waitKey(1) & 0xFF
    if k == 32:  #spacebar
        beginCollaboration(processed_frame)
    elif k == 27:  #esc
        stopCollaboration()
    elif k == 113:  #q
        break



# Release the VideoCapture object and close display windows
if useVid:
    vid.release()
cv.destroyAllWindows()