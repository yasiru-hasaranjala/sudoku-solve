import cv2
import numpy as np
from scipy import ndimage
import math
import sudokuSolver 
import copy


#converting cloured image to gray scale,blurring, thresholding, find contours, get biggeset contour, get corners and get perspective transform
def sudoku_recognition_solve(image, model, old_sudoku):

    clone_image = np.copy(image)      #cloning the image

    # converting to a gray image, blur that gray image for easier detectionand apply adaptiveThreshold
    gray_filter = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_filter = cv2.GaussianBlur(gray_filter, (5,5), 0)
    adaptive_thresh = cv2.adaptiveThreshold(blur_filter, 255, 1, 1, 11, 2)

    # finding all contours
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #extracting the contour with the biggest area
    max_area = 0
    biggest_contour = None
    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            biggest_contour = c

    if biggest_contour is None:        
        return image

    #getting the 4 corners of the biggeset contour
    corners = get_corners_from_contours(biggest_contour, 4)

    if corners is None:        
        return image


    #locating top left,top right,bottom left and bottom right corners
    rect = np.zeros((4, 2), dtype = "float32")
    corners = corners.reshape(4,2)

    # finding the top left sum
    sum = 10000
    index = 0
    for i in range(4):
        if(corners[i][0]+corners[i][1] < sum):
            sum = corners[i][0]+corners[i][1]
            index = i
    rect[0] = corners[index]
    corners = np.delete(corners, index, 0)

    # finding the bottom right sum
    sum = 0
    for i in range(3):
        if(corners[i][0]+corners[i][1] > sum):
            sum = corners[i][0]+corners[i][1]
            index = i
    rect[2] = corners[index]
    corners = np.delete(corners, index, 0)

    # finding the top right sum
    if(corners[0][0] > corners[1][0]):
        rect[1] = corners[0]
        rect[3] = corners[1]
        
    else:
        rect[1] = corners[1]
        rect[3] = corners[0]

    rect = rect.reshape(4,2)

    #A, B,C,D 4 corners selecting as an approximately corners of a square
    A = rect[0]
    B = rect[1]
    C = rect[2]
    D = rect[3]

    #if all 4 angles are not approximately 90 degrees return the image and stop
    AB = B - A      # 4 vectors AB AD BC DC
    AD = D - A
    BC = C - B
    DC = C - D
    eps_angle = 20
    if not (approx_90_degrees(angle_between(AB,AD), eps_angle) and approx_90_degrees(angle_between(AB,BC), eps_angle)
    and approx_90_degrees(angle_between(BC,DC), eps_angle) and approx_90_degrees(angle_between(AD,DC), eps_angle)):
        return image
    
    # the Lengths of AB, AD, BC, DC have to be approximately equal
    eps_scale = 1.2     # Longest cannot be longer than epsScale * shortest
    if(side_lengths_are_too_different(A, B, C, D, eps_scale)):
        return image


    # the width of the Sudoku board
    (tl, tr, br, bl) = rect
    width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # the height of the Sudoku board
    height_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))


    #taking the maximum height,width values to reach the final dimensions
    max_width = max(int(width_A), int(width_B))
    max_height = max(int(height_A), int(height_B))

   
    #constructing destination points
    destination = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype = "float32")

    #calculating the perspective transform matrix and warping the perspective to grab the screen
    perspective_transformed_matrix = cv2.getPerspectiveTransform(rect, destination)
    warp = cv2.warpPerspective(image, perspective_transformed_matrix, (max_width, max_height))
    orginal_warp = np.copy(warp)

    #getting  ready for recognizing digits
    warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)
    warp = cv2.GaussianBlur(warp, (5,5), 0)
    warp = cv2.adaptiveThreshold(warp, 255, 1, 1, 11, 2)
    warp = cv2.bitwise_not(warp)
    _, warp = cv2.threshold(warp, 150, 255, cv2.THRESH_BINARY)

    #initital grid to store sudoku board digits
    SIZE = 9
    grid = []
    for i in range(SIZE):
        row = []
        for j in range(SIZE):
            row.append(0)
        grid.append(row)

    height = warp.shape[0] // 9
    width = warp.shape[1] // 9

    offset_width = math.floor(width / 10)    # offset is used to get rid of the boundaries
    offset_height = math.floor(height / 10)

    #dividing the Sudoku board into 9x9 square:
    for i in range(SIZE):
        for j in range(SIZE):

            # crop with offset to avoid including boundaries
            cropped_image = warp[height*i+offset_height:height*(i+1)-offset_height, width*j+offset_width:width*(j+1)-offset_width]        
            
            #removing all the black lines near the edges
            ratio = 0.6        
            # Top
            while np.sum(cropped_image[0]) <= (1-ratio) * cropped_image.shape[1] * 255:
                cropped_image = cropped_image[1:]
            # Bottom
            while np.sum(cropped_image[:,-1]) <= (1-ratio) * cropped_image.shape[1] * 255:
                cropped_image = np.delete(cropped_image, -1, 1)
            # Left
            while np.sum(cropped_image[:,0]) <= (1-ratio) * cropped_image.shape[0] * 255:
                cropped_image = np.delete(cropped_image, 0, 1)
            # Right
            while np.sum(cropped_image[-1]) <= (1-ratio) * cropped_image.shape[0] * 255:
                cropped_image = cropped_image[:-1]    

            #taking the largest digit  and removing all the noises
            cropped_image = cv2.bitwise_not(cropped_image)
            cropped_image = largest_connected_component(cropped_image)
           
            # resizing
            size_of_digit = 28
            cropped_image = cv2.resize(cropped_image, (size_of_digit,size_of_digit))

           
            #if there is a white cell it has little blackp pixels.So if there is a wite cell grid[i][j]==0 and continue
            if cropped_image.sum() >= size_of_digit**2*255 - size_of_digit * 1 * 255:
                grid[i][j] == 0
                continue     

            #if there is a huge white area in the center
            center_width = cropped_image.shape[1] // 2
            center_height = cropped_image.shape[0] // 2
            x_start = center_height // 2
            x_end = center_height // 2 + center_height
            y_start = center_width // 2
            y_end = center_width // 2 + center_width
            center_region = cropped_image[x_start:x_end, y_start:y_end]
            
            if center_region.sum() >= center_width * center_height * 255 - 255:
                grid[i][j] = 0
                continue    #continuing of there is a white cell

            # string the number of rows and cols
            rows, cols = cropped_image.shape


            #applying binary threshold to make digits more clear
            _, cropped_image = cv2.threshold(cropped_image, 200, 255, cv2.THRESH_BINARY) 
            cropped_image = cropped_image.astype(np.uint8)

            # centralizing the image according to center of mass
            cropped_image = cv2.bitwise_not(cropped_image)
            shift_x, shift_y = get_best_shift(cropped_image)
            shifted = shift(cropped_image,shift_x,shift_y)
            cropped_image = shifted

            cropped_image = cv2.bitwise_not(cropped_image)
            
            # converting to a proper format to recognize
            cropped_image = prepare(cropped_image)

            # recognizing digits
            prediction = model.predict([cropped_image]) 
            grid[i][j] = np.argmax(prediction[0]) + 1            # 1 2 3 4 5 6 7 8 9 starts from 0, so add 1

    user_grid = copy.deepcopy(grid)

    #printing the same solution if the same board is used for last camera frame
    if (not old_sudoku is None) and two_matrices_are_equal(old_sudoku, grid, 9, 9):
        if(sudokuSolver.all_board_non_zero(grid)):
            orginal_warp = write_solution_on_image(orginal_warp, old_sudoku, user_grid)
  
    #solving the sudoku if this is a different board
    else:
        sudokuSolver.solve_sudoku(grid)                
        if(sudokuSolver.all_board_non_zero(grid)):    
            orginal_warp = write_solution_on_image(orginal_warp, grid, user_grid)
            old_sudoku = copy.deepcopy(grid)      # keepiong the copy of old solution

    # applying the inverse perspective transform and paste the solutions on top of the orginal image
    sudoku_result = cv2.warpPerspective(orginal_warp, perspective_transformed_matrix, (image.shape[1], image.shape[0])
                                        , flags=cv2.WARP_INVERSE_MAP)
    result = np.where(sudoku_result.sum(axis=-1,keepdims=True)!=0, sudoku_result, image)

    return result


# writing the solution on image frame
def write_solution_on_image(image, grid, user_grid):
    # writing the grid on image
    SIZE = 9
    width = image.shape[1] // 9
    height = image.shape[0] // 9
    for i in range(SIZE):
        for j in range(SIZE):
            if(user_grid[i][j] != 0):    # if user fills this cell
                continue                # move on
            text = str(grid[i][j])
            off_set_x = width // 15
            off_set_y = height // 15
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_height, text_width), baseLine = cv2.getTextSize(text, font, fontScale=1, thickness=3)
            margin_X = math.floor(width / 7)
            margin_Y = math.floor(height / 7)
        
            font_scale = 0.6 * min(width, height) / max(text_height, text_width)
            text_height *= font_scale
            text_width *= font_scale
            bottom_left_corner_x = width*j + math.floor((width - text_width) / 2) + off_set_x
            bottom_left_corner_y = height*(i+1) - math.floor((height - text_height) / 2) + off_set_y
            image = cv2.putText(image, text, (bottom_left_corner_x, bottom_left_corner_y), 
                                                  font, font_scale, (0,0,255), thickness=3, lineType=cv2.LINE_AA)
    return image

# comparing every single elements of 2 matrices and returning if all corresponding entries are equal
def two_matrices_are_equal(matrix_1, matrix_2, row, col):
    for i in range(row):
        for j in range(col):
            if matrix_1[i][j] != matrix_2[i][j]:
                return False
    return True


#detecting whether the image frame is a sudoku board or not using the length of the sides
def side_lengths_are_too_different(A, B, C, D, eps_scale):
    AB = math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
    AD = math.sqrt((A[0]-D[0])**2 + (A[1]-D[1])**2)
    BC = math.sqrt((B[0]-C[0])**2 + (B[1]-C[1])**2)
    CD = math.sqrt((C[0]-D[0])**2 + (C[1]-D[1])**2)
    shortest = min(AB, AD, BC, CD)
    longest = max(AB, AD, BC, CD)
    return longest > eps_scale * shortest


#finding all 4 angles are approximately 90 degrees with tolerance 
def approx_90_degrees(angle, e):
    return abs(angle - 90) < e


#seperating the digit from the nois in the cropped image which is 9x9 small square images from the sudoku board
def largest_connected_component(image):

    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[:, -1]

    if(len(sizes) <= 1):
        blank_image = np.zeros(image.shape)
        blank_image.fill(255)
        return blank_image

    max_label = 1
    #starting from component 1  because we want to leave out the background
    max_size = sizes[1]     

    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img_2 = np.zeros(output.shape)
    img_2.fill(255)
    img_2[output == max_label] = 0
    return img_2

#returning the angle between 2 vectors in degrees
def angle_between(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector2)
    angle = np.arccos(dot_product)
    return angle * 57.2958                                       # converting to degrees

# calculations for centralizing the image using its center of mass
def get_best_shift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    return shiftx, shifty

#shifting the image using the return variables of get_best_shift function
def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted


#getting the 4 corners from contour
def get_corners_from_contours(contours, corner_amount=4, max_iterations=200):

    coefficient = 1
    while max_iterations > 0 and coefficient >= 0:
        max_iterations = max_iterations - 1

        e = coefficient * cv2.arcLength(contours, True)

        poly_approx = cv2.approxPolyDP(contours, e, True)
        hull = cv2.convexHull(poly_approx)
        if len(hull) == corner_amount:
            return hull
        else:
            if len(hull) > corner_amount:
                coefficient += .01
            else:
                coefficient -= .01
    return None

#preparing and normalizing the image to get ready for digit recognition
def prepare(img_array):
    new_array = img_array.reshape(-1, 28, 28, 1)
    new_array = new_array.astype('float32')
    new_array /= 255
    return new_array

def showImage(img, name, width, height):
    new_image = np.copy(img)
    new_image = cv2.resize(new_image, (width, height))
    cv2.imshow(name, new_image)
