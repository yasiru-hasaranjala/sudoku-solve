
import cv2
import numpy as np
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import RealTimeSudokuSolver
import sudokuSolver
from scipy import ndimage
import math
import copy


def showImage(img, name, width, height):
    new_image = np.copy(img)
    new_image = cv2.resize(new_image, (width, height))
    cv2.imshow(name, new_image)

# Load and set up Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)    # HD Camera
cap.set(4, 720)

# Loading model (Load weights and configuration seperately to speed up model and predictions)
input_shape = (28, 28, 1)
num_classes = 9
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Load weights from pre-trained model. This model is trained in digitRecognition.py
model.load_weights("digitRecognition.h5")   

old_sudoku = None # Used to compare new sudoku or old sudoku
while(True):
    ret, frame = cap.read() # Read the frame
    if ret == True:
        
        # RealTimeSudokuSolver.recognize_sudoku_and_solve
        sudoku_frame = RealTimeSudokuSolver.recognize_sudoku_and_solve(frame, model, old_sudoku) 
        showImage(sudoku_frame, "Real Time Sudoku Solver", 1066, 600)
        if cv2.waitKey(1) == ord('q'):   # Hit q if you want to stop the camera
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()