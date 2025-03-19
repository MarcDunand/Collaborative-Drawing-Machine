import cv2
import numpy as np
import os
from scipy.ndimage import convolve
from dataclasses import dataclass
import random as rand


@dataclass
class Island:
    coord : tuple[int, int]
    size : int
    points : np.array
    surfaces : list
    roofs : list



#generates a random rgb color
def randColor(min, max):
    return [rand.randrange(min, max), rand.randrange(min, max), rand.randrange(min, max)]

#Creates an image array with ground as black and everything else as white
def generate_detected_ground(binary_array):
    # Convert 0s to 255 and 1s to 0 efficiently
    binArrFlipped = np.where(binary_array == 0, 255, 0).astype(np.uint8)
    return binArrFlipped


#converts an image array and saves the resulting image
def display_image(file, filename, output_folder, process_img = True):
    if process_img:
        file = generate_detected_ground(file)

    name, ext = os.path.splitext(filename)
    output_filename = f"{name}_output{ext}"
    output_path = os.path.join(output_folder, output_filename)

    cv2.imwrite(output_path, file)


#Creates an image array with ground encoded as 1s and empty space encoded as 0s
def image_to_binary_array(image_path, threshold=128):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Apply thresholding
    binary_array = (image < threshold).astype(np.uint8)
    
    return binary_array


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


def fill_holes_morph(binary_image):
    kernel = np.ones((5,5), np.uint8)  # Small structuring element
    closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    return closed


def find_surfaces(island):
    points = island.points  # Assuming this is an Nx2 NumPy array

    # Convert to a set of tuples for fast lookups
    points_set = {tuple(p) for p in points}

    # Find surface points (points with no direct neighbor above)
    surfacePoints = np.array([p for p in points if (p[0] - 1, p[1]) not in points_set])

    # Sort first by x, then by y (to process left-to-right)
    surfacePoints = surfacePoints[np.lexsort((surfacePoints[:, 0], surfacePoints[:, 1]))]

    surfacePoints = [np.array(point) for point in surfacePoints] 

    surfaces = []
    while len(surfacePoints) > 0:
        prev = surfacePoints.pop(0)
        surface = [prev]
        i = 0
        endSurface = False
        while(i < len(surfacePoints) and not endSurface):
            cur = surfacePoints[i]
            if prev[1] == cur[1]:  #if X has not increased, keep going until X increases
                i += 1
            elif prev[1] + 1 == cur[1]:  #if X has increased by 1, we check if connected
                if abs(prev[0] - cur[0]) <= 20:  #checks if connected
                    surface.append(surfacePoints.pop(i))
                    prev = cur
                else:
                    i += 1  #increment if we didn't pop an element from the list
            elif prev[1] + 1 < cur[1]:  #if X has incremented by more than 1, the surface is not connected
                if len(surface) > 10:
                    surfaces.append(np.array(surface))
                endSurface = True
                surface = []

        if len(surface) > 0:
            surfaces.append(np.array(surface))

                
    return surfaces


def find_area_height(image, surface):
    roof = np.empty(surface.shape, dtype=surface.dtype)
    for i in range(len(surface)):
        (y, x) = surface[i]
        # Extract column up to y (excluding y itself)
        column_above = image[:y, x]

        # Find row indices where the pixel is black (0)
        black_pixel_rows = np.where(column_above == 1)[0]

        if black_pixel_rows.size > 0:
            roof[i] = (black_pixel_rows[-1] + 1, x)  # Last black pixel in the search direction (top-down)
        else:
            roof[i] = (0, x)
    
    return roof


def main(frame, thresh, imageProvided=True):
    #defines where the island visualizer image will be stored
    output_folder = "sampleImages_output"
    os.makedirs(output_folder, exist_ok=True)

    if imageProvided:
        binary_array = (frame < thresh).astype(np.uint8)
    else:
        # Define the path to the image
        input_folder = "sampleImages"
        input_filename = "squiggles.jpg"

        # Convert image to binary array
        binary_array = image_to_binary_array(os.path.join(input_folder, input_filename), thresh)
    
    
    binary_array_H, binary_array_W = binary_array.shape


    # Fill in any pixel that is surrounded by 3 or more filled pixels
    filled_holes_img = fill_enclosed_pixels(binary_array)

    # Use a morphological dialation/erosion system to close holes
    morpho_img = fill_holes_morph(filled_holes_img)


    #identify the groups of black pixels, "islands"
    num_labels, labeled_img, stats, centroids = cv2.connectedComponentsWithStats(morpho_img, connectivity=4)

    islandList = []
    for label in range(1, num_labels):  # Start from 1 to skip background
        pixel_coords = np.column_stack(np.where(labeled_img == label))
        islandList.append(Island(pixel_coords[0], stats[label, cv2.CC_STAT_AREA], pixel_coords, [], []))

    islandImg = np.zeros((binary_array_H, binary_array_W, 3), dtype=np.uint8)

    #remove islands that are too small
    islandList = [island for island in islandList if island.size >= 200]


    #find surfaces on islands
    for island in islandList:
        island.surfaces = find_surfaces(island)
        for surface in island.surfaces:
            island.roofs.append(find_area_height(morpho_img, surface))

        if len(island.surfaces) != len(island.roofs):
            print("ERROR: unequal number of surfaces and roofs")
            


    #create visualization of processed image
    for island in islandList:
        islandImg[island.points[:, 0], island.points[:, 1]] = randColor(20, 255)
        for surface in island.surfaces:
            islandImg[surface[:, 0], surface[:, 1]] = randColor(20, 255)
        for roof in island.roofs:
            islandImg[roof[:, 0], roof[:, 1]] = randColor(20, 255)

        for i in range(len(island.surfaces)):
            surface, roof = island.surfaces[i], island.roofs[i]
            
            y_indices = np.arange(islandImg.shape[0])[:, None]  # Column vector for row indices
            x_indices = surface[:, 1]  # Extract x-coordinates

            # Generate row indices where y is between roof and floor
            valid_rows = (y_indices > roof[:, 0]) & (y_indices < surface[:, 0])

            # Extract row indices and corresponding x indices
            row_indices, col_indices = np.where(valid_rows)  # Get valid (y, x) pairs

            # Apply the color to valid pixels
            islandImg[row_indices, x_indices[col_indices]] = randColor(140, 190)  # Properly mapped
            

    # Display sample image
    display_image(binary_array, "squiggles.jpg", output_folder)
    
    display_image(morpho_img, "squigglesMorph.jpg", output_folder)

    display_image(islandImg, "islands.jpg", output_folder, False)


    return islandList 
    