import cv2 as cv
import numpy as np
import glob
import os


images = [] # List to store images
imgpoints = [] #List to store checker points
objpoints = [] #List to store the real positions of the checker corners
patternSize = (9, 7) #the inner corners of the checkerboard
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.01)  #the algorithm will stop after 50 iterations or if the corner moves by less than 0.01 pixels, whichever comes first.
objp = np.zeros((1, patternSize[0] * patternSize[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2)


# Directory containing your images
image_folder = os.path.join(os.getcwd(), 'checkerboardsv2')

# Use glob to find all images in the specified folder
image_files = glob.glob(os.path.join(image_folder, '*.jpg'))


# Loop over the image files and read them
for file in image_files:
    img = cv.imread(file)  # Read the image using OpenCV
    if img is not None:
        images.append(img)  # Append the image to the list

#convert the list of images to a NumPy array
images_array = np.array(images)

#find corners
for i, imgColor in enumerate(images_array):
    img = cv.cvtColor(imgColor, cv.COLOR_BGR2GRAY)  #converts to greyscale
    
    retval, corners = cv.findChessboardCorners(img, patternSize)  #finds corners with lower accuracy
    
    print(retval)  #tells us if it works or not
    if retval:  #if it worked, continue
        objpoints.append(objp)
        cv.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)  #finds corners with high accuracy
        imgpoints.append(corners)
        
        imgColor = cv.drawChessboardCorners(imgColor, patternSize, corners, retval)  #draw the checkerboard
    #cv.imshow(f'Image {i+1}', imgColor)
    #cv.waitKey(0)  # Press any key to continue to the next image

cv.destroyAllWindows

h,w = img.shape[:2]

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
print(" \nAccuracy: ")
print(ret)
print(" \nCamera matrix :")
print(mtx)
print("\ndist coefficients: ")
print(dist)
# print("\nrvecs : ")
# print(rvecs)
# print("\ntvecs : ")
# print(tvecs)


#save matrix and dist coefficients
np.savez('calibration_data.npz', cameraMatrix=mtx, distCoeffs=dist)