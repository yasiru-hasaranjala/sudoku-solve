import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

heightImg = 450
widthImg = 450
model = load_model('Digit_Recognizer.h5')

#image preprocessing
def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                         # converting image to grayscale
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)                          #adding gaussian blut
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)         # applying adaptive threshold
    return imgThreshold


#re-ordering points for warp perspective
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


#finding the biggest contour which is the sudoku puzzle
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area


#splitting the inage into 81 different images
def splitBoxes(img):
    rows = np.vsplit(img,9)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (16,16), input_shape=(28, 28, 1), activation='relu'))
    model.add(Conv2D(8, (8,8), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(units=11, activation='softmax'))
    return model

#getting predictions on all images
def getPredection(boxes,model):
    result = []
    for image in boxes:

        #preparing the image
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] -4]
        img = cv2.resize(img, (28, 28))
        img = img / 255
        img = img.reshape(1, 28, 28, 1)
    
        #getting predictions
        predictions = model.predict(img)
        classIndex = model.predict_classes(img)
        probabilityValue = np.amax(predictions)
        print(classIndex,probabilityValue)

        #saving to results
        if probabilityValue > 0.8:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result

#displaying the soultion on the image
def display_numbers(img,numbers,color = (0,255,0)):
    section_width = int(img.shape[1]/9)
    section_height = int(img.shape[0]/9)
    for x in range (0,9):
        for y in range (0,9):
            if numbers[(y*9)+x] != 0 :
                 cv2.putText(img, str(numbers[(y*9)+x]),
                               (x*section_width+int(section_width/2)-10, int((y+0.8)*section_height)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
    return img

def find_empty(bo):
    for i in range(len(bo)):
        for j in range(len(bo[0])):
            if bo[i][j] == 0:
                return (i, j)  # row, col
    return None


def valid(bo, num, pos):
    #checking row
    for i in range(len(bo[0])):
        if bo[pos[0]][i] == num and pos[1] != i:
            return False
    #checking column
    for i in range(len(bo)):
        if bo[i][pos[1]] == num and pos[0] != i:
            return False
    #checking box
    box_x = pos[1] // 3
    box_y = pos[0] // 3
    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if bo[i][j] == num and (i,j) != pos:
                return False
    return True


def solve(bo):
    find = find_empty(bo)
    if not find:
        return True
    else:
        row, col = find
    for i in range(1,10):
        if valid(bo, i, (row, col)):
            bo[row][col] = i
            if solve(bo):
                return True
            bo[row][col] = 0
    return False

#deawing the grid to see the warp perspective efficiency
def drawGrid(img):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for i in range (0,9):
        pt1 = (0,secH*i)
        pt2 = (img.shape[1],secH*i)
        pt3 = (secW * i, 0)
        pt4 = (secW*i,img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0),2)
        cv2.line(img, pt3, pt4, (255, 255, 0),2)
    return img

#finding the biggest contour and usig it as sudoku board
def solveByImage(pathImage = "test.jpg"):
    print(pathImage)
    #preparing the image
    img = cv2.imread(pathImage)
    img = cv2.resize(img, (widthImg, heightImg))  #resizing the image to make a square image
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # creating a blank image for testing purposes
    imgThreshold = preProcess(img)

    print(imgThreshold)


    #finding all contours
    imgContours = img.copy() 
    imgBigContour = img.copy() 
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #finding all contours
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)      #drawing all detected contours


    print(imgContours)

    biggest, maxArea = biggestContour(contours)     #finding the biggest contour 
    print(biggest)
    if biggest.size != 0:
        biggest = reorder(biggest)
        print(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25)  #drawing the biggest contour
        pts1 = np.float32(biggest) 
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])     #preparing points for warp
        matrix = cv2.getPerspectiveTransform(pts1, pts2) 
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
        imgDetectedDigits = imgBlank.copy()
        imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)


       #splitting the image and finding each digit available
        imgSolvedDigits = imgBlank.copy()
        boxes = splitBoxes(imgWarpColored)
        print(len(boxes))

        numbers = getPredection(boxes, model)
        print(numbers)
        imgDetectedDigits = display_numbers(imgDetectedDigits, numbers, color=(255, 0, 255))
        numbers = np.asarray(numbers)
        posArray = np.where(numbers > 0, 0, 1)
        print(posArray)


       #finding the solution of the board
        board = np.array_split(numbers,9)
        print(board)
        try:
            solve(board)
        except:
            pass
        print(board)
        flatList = []
        for sublist in board:
            for item in sublist:
                flatList.append(item)
        solvedNumbers =flatList*posArray
        imgSolvedDigits= display_numbers(imgSolvedDigits,solvedNumbers)


        #overlaying the solution
        pts2 = np.float32(biggest) #preparing points for warp
        pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) #preparing points for warp
        matrix = cv2.getPerspectiveTransform(pts1, pts2)  
        imgInvWarpColored = img.copy()
        imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
        inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
        imgDetectedDigits = drawGrid(imgDetectedDigits)
        imgSolvedDigits = drawGrid(imgSolvedDigits)

        cv2.imwrite("solved.jpg",inv_perspective)

    else:
        print("No Sudoku Found")

cv2.waitKey(0)



