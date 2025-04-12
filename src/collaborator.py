import cv2 as cv
import numpy as np
import random as rand
from pyaxidraw import axidraw
from scipy.ndimage import convolve
import argparse
import imageConverter as imgc
from imageConverter import Island
import axiDoodles as do
import previewDoodles as pr
import featurePlotting as plot
import threading
import time
from PIL import Image, ImageDraw
import traceback
import serial




#Defines whether the physical button is pressed or not
class ButtonState:
    def __init__(self):
        self.pressed = False



#Arduino Values
arduino_port = 'COM8'  #change to match usb port
baud = 9600

# Define the initial variables
xDef = 4656  #resolution of the camera
yDef = 3496

thresh = 157  #higher means more land

#controls the size of the cropped in image
cropXmin = 1055
cropYmin = 1615
cropW = 1354
cropH = 965

S = 454
xc = 118
yc = 115
isDrawing = False  #True when the Axidraw is running
isRunning = False  #True when runCollaboration thread is running

useVid = True  # Choose whether to use camera or internal file for capture
useAxi = True  # For bugfixing while away from axidraw, program only works correctly with val is True



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
    xc = val/10 - 10

def on_ystrackbar(val):
    global yc
    yc = val/10 - 10

def on_Strackbar(val):
    global S
    S = val/2000






#Get input from "go" button
def listen_to_arduino(buttonState):
    while True:
        line = ser.readline().decode('utf-8').strip()
        if line == "Button Pressed!":
            buttonState.pressed = True
        else:
            buttonState.pressed = False



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


def plotReceipt(receipt):
    global isRunning

    for subReceipt in receipt:
        for feature in subReceipt:
            #Can interrupt drawing by pressing esc
            if not isRunning:
                print("Drawing interrupted!")
                axi.moveto(0, 0)
                return
            
            #attempt to draw a feature
            try: 
                if feature[0] == "S":  #striation
                    plot.striation(axi, feature[1], S, xc, yc)
                elif feature[0] == "B":  #Bird
                    plot.bird(axi, feature[1], S, xc, yc)
                elif feature[0] == "Tr":  #Tree
                    plot.tree(axi, feature[1], S, xc, yc)
                elif feature[0] == "To":  #Tower
                    plot.tower(axi, feature[1], S, xc, yc)
                elif feature[0] == "L":  #Lake
                    plot.lake(axi, feature[1], S, xc, yc)

            except Exception as e:
                print(f"Error during plotting: {e}")
                traceback.print_exc()
                axi.penup()

    axi.moveto(0, 0)
                

def previewSurface(draw, tracedLine, overhang):
    global isDrawing
    isDrawing = True

    subReceipt = []  #holds all data about all features drawn
    birdScale = 3

    d = 0  #keeps track of the closest position a new feature can be drawn
    for i in range(len(tracedLine)):
        if i > d:
            [y, x] = tracedLine[i]
            [hy, _] = overhang[i]
            h = y - hy  #height on this column of pixels

            #attempts to generate altitude striations
            if rand.random() < 0.1:
                striaInfo = pr.drawStriation(draw, tracedLine, i, x, y, 10, 80)

                if striaInfo != -1:  #if a line has been drawn
                    subReceipt.append(("S", striaInfo))  #add info about the drawn striation to receipt

            #generates single birds
            if rand.random() < 0.01 and y > 2*birdScale:
                birdy = rand.uniform(10, (y-3*birdScale))
                pr.drawBird(draw, x, birdy, birdScale, 60)

                subReceipt.append(("B", [x, birdy, birdScale, 60]))  #add drawn bird to receipt

            #generate flocks    
            if rand.random() < 0.003 and y > 2+3*birdScale:
                stepSize = birdScale*10
                birdx = x
                birdy = rand.uniform(10, (y-3*birdScale))
                for i in range(rand.randrange(5, 30)):
                    birdx += rand.uniform(-1*stepSize, stepSize)
                    birdy += rand.uniform(-1*stepSize*S, stepSize*S)
                    pr.drawBird(draw, birdx, birdy, birdScale, 60)

                    subReceipt.append(("B", [birdx, birdy, birdScale, 60]))  #add drawn bird to receipt


            featureGen = rand.random()
            if featureGen < 0.1:  # 10% tree
                (d, treeInfo) = pr.drawTree(draw, i, x, y, min(60, h))
                subReceipt.append(("Tr", treeInfo))

            elif featureGen < 0.115: # 1.5% tower
                (d, towerInfo)  = pr.drawTower(draw, tracedLine, overhang, i)
                if towerInfo != -1:
                    subReceipt.append(("To", towerInfo))

            elif featureGen < 0.12: # 0.5% village
                (d, villageInfo) = pr.drawVillage(draw, tracedLine, overhang, i)
                subReceipt.extend(villageInfo)

            elif featureGen < 0.17: # 5% lake
                (d, lakeInfo) = pr.drawLake(draw, tracedLine, i, x, y, 20, 150, 6)
                if lakeInfo != -1:
                    subReceipt.append(("L", lakeInfo))

    return subReceipt




#ONGOING to be used to draw features that are not surface-depenedent (like bridges)
def previewSky(draw, surfaceList):
    for i in range(len(surfaceList)):
        s1 = surfaceList[i]
        [s1xL, s1yL] = s1[0]
        [s1xR, s1yR] = s1[-1]
        print(type(s1))
        for j in range(i+1, len(surfaceList)):
            s2 = surfaceList[j]
            [s2xL, s2yL] = s2[0]
            [s2xR, s2yR] = s2[-1]

            





# Test to see if new surface detection is working
def traceSurfaces(islandList):
    global isRunning
    
    axi.penup()
    for island in islandList:
        for surface in island.surfaces:
            [startY, startX] = surface[0]
            axi.penup()
            axi.moveto(startX*S+xc, startY*S+yc)
            axi.pendown()

            for i in range(0, len(surface)):
                if not isRunning:
                    print("Drawing interrupted!")
                    axi.moveto(0, 0)
                    return
                
                [y, x] = surface[i]
                axi.lineto(x*S+xc, y*S+yc)
                print(x, y)


            axi.penup()

    axi.moveto(0, 0)


# Begins the collaboration, gets island information from the raw image
def runCollaboration(frame):
    global isRunning

    # Runs imageConverter to get data about islands and surfaces in the drawn image
    islandList = imgc.main(frame)
    surfaceList = []
    for island in islandList:
        surfaceList.extend(island.surfaces)

    #setup preview image file
    greyscaleConvert = np.where(frame == 0, 255, 0).astype(np.uint8)
    preview = Image.fromarray(greyscaleConvert).convert("L")
    draw = ImageDraw.Draw(preview)

    #create preview image file and calculate features to be drawn
    receipt = []
    #receipt.append(previewSky(draw, surfaceList))  #TODO impliment
    for island in islandList:
        for i in range(len(island.surfaces)):
            surface = island.surfaces[i]
            overhang = island.overhangs[i]
            receipt.append(previewSurface(draw, surface, overhang))
    
    preview.save("./sampleImages_output/collaboration_preview.jpg")

    #Draw features with axi
    if useAxi:
        #traceSurfaces(islandList)  # Test attempting to trace the location of all island surfaces
        try:
            plotReceipt(receipt)
        except Exception as e:
            print(f"Error during plotting: {e}")
            traceback.print_exc()
            axi.goto(0, 0)
        else:
            print("Plotting succesful!")
        
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



#Connect to arduino
ser = serial.Serial(arduino_port, baud)
time.sleep(2)
buttonState = ButtonState()  #access to whether the physical button has been pressed
threading.Thread(target=listen_to_arduino, args=(buttonState,), daemon=True).start()
print("Connected to Arduino!")



#Sets up image source
if useVid:
    # Define a video capture object
    vid = cv.VideoCapture(0, cv.CAP_DSHOW)
    vid.set(cv.CAP_PROP_FRAME_WIDTH, xDef)
    vid.set(cv.CAP_PROP_FRAME_HEIGHT, yDef)
    
    if not vid.isOpened():
        raise IOError("Cannot open webcam")
    
    vid.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)
    vid.set(cv.CAP_PROP_EXPOSURE, -8)
else:
    # Gets the image file
    parser = argparse.ArgumentParser(description='Code for Finding contours in your image tutorial.')
    parser.add_argument('--input', help='', default='./sampleImages/bwWiggles.jpg')
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



#Times the framerate
last_capture_time = 0
capture_interval = 1



#execution loop
while True:
    # Get the frame that will be used to create the collaboration
    if useVid and time.time() - last_capture_time > capture_interval:
        last_capture_time = time.time()

        ret, frame = vid.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        fullHeight, fullWidth = frame.shape[:2]
        yMin = fullHeight - cropYmin

        #gets the proper crop and rotation
        frame = cv.resize(frame[cropXmin:cropXmin+cropW, yMin:yMin+cropH], (cropH, cropW))
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)

    elif not useVid:
        frame = src
        
    #generates the frame that will be interpreted by the collaborator
    processed_frame = preprocess_image(frame, thresh)

    #scales preview to desired window size
    windowWidth = 1000
    croppedHeight, croppedWidth = frame.shape[:2]
    scale = windowWidth / croppedWidth
    new_w = windowWidth
    new_h = int(croppedHeight * scale)
    display_frame = cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_AREA)

    #shows live image
    cv.imshow('Positioning', display_frame)

    #shows the frame as it will be interpreted by the collaborator
    cv.imshow('Contours and Correction', np.where(processed_frame == 0, 255, 0).astype(np.uint8))

    # Key commands
    k = cv.waitKey(1) & 0xFF
    if k == 32 or buttonState.pressed:  #spacebar
        beginCollaboration(processed_frame)
    elif k == 27:  #esc
        stopCollaboration()
    elif k == 113:  #q
        break



# Release the VideoCapture object and close display windows
if useVid:
    vid.release()
cv.destroyAllWindows()