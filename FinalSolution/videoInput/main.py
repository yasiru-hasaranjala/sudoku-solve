
import cv2
import numpy as np
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

# load and set up the camera frames
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)    # HD Camera
cap.set(4, 720)


input_shape = (28, 28, 1)
num_classes = 9

#loading weights and configurations seperately to speed up model and predictions

def build_model():    
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
    model.load_weights("digitRecognition.h5")                #loading weights from the per-trained model which was trained in digitRecognition.py

    return model

model=build_model()



old_sudoku = None                   #will be used to compare new sudoku or old sudoku
while(True):
    ret, frame = cap.read() # Read the frame
    if ret == True:        
        # recognizing sudoku in real time and solve
        sudoku_frame = RealTimeSudokuSolver.sudoku_recognition_solve(frame, model, old_sudoku) 
        showImage(sudoku_frame, "Real Time Sudoku Solver", 1066, 600)
        if cv2.waitKey(1) == ord('q'):   # "q" will be used as the exit key
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()