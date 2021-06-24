import sys
from tkinter import * #or Tkinter if you're on Python2.7
from PIL import Image, ImageTk
import tkinter.filedialog as tkFileDialog
import sudoku
from videoInput.videoMode import *
from sudoku import *
import ntpath
import os
imgPath = ""

def loadImg(window2_main):
    path=tkFileDialog.askopenfilename(filetypes=[("Image File",'.jpg')])
    im = Image.open(path)
    im.save("test.jpg")
    imgPath = "test.jpg"
    width, height = im.size
    im = im.resize((400, int(height*400/width)), Image.ANTIALIAS)
    tkimage = ImageTk.PhotoImage(im)
    myvar=Label(window2_main,image = tkimage)
    myvar.image = tkimage
    myvar.grid(row=2,column=1, padx=(0, 0), pady=(0, 15), columnspan=4, sticky="nsew")

def solve(path, window2_main):
    solveByImage(path)
    imtest =Image.open('test.jpg')
    width, height = imtest.size
    im = Image.open('solved.jpg')
    im = im.resize((400, int(height*400/width)), Image.ANTIALIAS)
    tkimage = ImageTk.PhotoImage(im)
    myvar=Label(window2_main,image = tkimage)
    myvar.image = tkimage
    myvar.grid(row=2,column=5, padx=(0, 0), pady=(0, 15), columnspan=4, sticky="nsew")

def loadVdo(window3_main):
    print("dfsfd")
    
root = Tk()
def window1(window2_main):
    window2_main.destroy()
    window1_main = Tk()
    window1_main.title("Sudoku Solver")
    lable1 = Label(window1_main, text="Sudoku Solver", font=("Courier", 25))
    button1 = Button(window1_main,text ='Solve by Image',command = lambda: window2(window1_main), height=2, width=20, highlightbackground="#37d3ff")
    button2 = Button(window1_main,text ='Solve by Video',command = lambda: window3(window1_main), height=2, width=20)
    button3 = Button(window1_main,text ='Close',command = window1_main.destroy, height=2, width=20)
    lable1.grid(row=0, column=0, padx=(70, 70), pady=(20, 20))
    button1.grid(row=1, column=0, padx=(70, 70), pady=(2, 20))
    button2.grid(row=2, column=0, padx=(70, 70), pady=(20, 20))
    button3.grid(row=3, column=0, padx=(70, 70), pady=(20, 20))
    window1_main.mainloop()

def window2(window1_main):
    window1_main.destroy()
    window2_main = Tk()
    window2_main.title("Solve by Image")
    lable1 = Label(window2_main, text="Solve by Image", font=("Courier", 25))
    button1 = Button(window2_main,text ='Open Image',command = lambda: loadImg(window2_main), height=2, width=20)
    button2 = Button(window2_main,text ='Solve',command = lambda: solve("test.jpg", window2_main), height=2, width=20)
    button3 = Button(window2_main,text ='Close',command = lambda: window1(window2_main), height=2, width=20)
    lable2 = Label(window2_main, text="")
    lable1.grid(row=0, column=0, columnspan=8, padx=(70, 70), pady=(10, 0))
    button1.grid(row=1, column=3, padx=(175, 20), pady=(20, 20))
    button2.grid(row=1, column=4, padx=(20, 20), pady=(20, 20), columnspan=2)
    button3.grid(row=1, column=6, padx=(20, 175), pady=(20, 20))
    lable2.grid(row=2, column=0, pady=(200, 200))
    window2_main.mainloop()

def window3(window1_main):
    window1_main.destroy()
    window3_main = Tk()
    window3_main.title("Solve by Video")
    lable1 = Label(window3_main, text="Solve by Video", font=("Courier", 25))
    button1 = Button(window3_main,text ='Open Video',command = videoInput.videoMode.videoView, height=2, width=20)
    button2 = Button(window3_main,text ='Close',command = lambda: window1(window3_main), height=2, width=20)
    lable1.grid(row=0, column=0, columnspan=2, padx=(70, 70), pady=(20, 20))
    button1.grid(row=1, column=0, padx=(20, 20), pady=(20, 20))
    button2.grid(row=1, column=1, padx=(20, 20), pady=(20, 20))
    window3_main.mainloop()

window1(root)
root.mainloop()
