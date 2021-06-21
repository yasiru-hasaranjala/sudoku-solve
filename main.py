import sys
from tkinter import * #or Tkinter if you're on Python2.7
from PIL import Image, ImageTk
import tkinter.filedialog as tkFileDialog
import sudoku
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
    im = im.resize((int(width*450/height) ,450), Image.ANTIALIAS)
    tkimage = ImageTk.PhotoImage(im)
    myvar=Label(window2_main,image = tkimage)
    myvar.image = tkimage
    myvar.grid(row=1,column=0, columnspan=3, sticky="nsew")

def solve(path, window2_main):
    solveByImage(path)
    im = Image.open('solved.jpg')
    width, height = im.size
    im = im.resize((int(width*450/height) ,450), Image.ANTIALIAS)
    tkimage = ImageTk.PhotoImage(im)
    myvar=Label(window2_main,image = tkimage)
    myvar.image = tkimage
    myvar.grid(row=1,column=0, columnspan=3, sticky="nsew")


root = Tk()
root.title("Sudoku Solver")
def window1(window2_main):
    window2_main.destroy()
    window1_main = Tk()
    window1_main.title("Sudoku Solver")
    button1 = Button(window1_main,text ='Solve by Image',command = lambda: window2(window1_main), height=2, width=20)
    button2 = Button(window1_main,text ='Solve by Video',command = lambda: solve("test.jpg"), height=2, width=20)
    button3 = Button(window1_main,text ='Close',command = window1_main.destroy, height=2, width=20)
    button1.grid(row=0, column=0, padx=(20, 20), pady=(20, 20))
    button2.grid(row=1, column=0, padx=(20, 20), pady=(20, 20))
    button3.grid(row=2, column=0, padx=(20, 20), pady=(20, 20))
    window1_main.mainloop()

def window2(window1_main):
    window1_main.destroy()
    window2_main = Tk()
    window1_main.title("Solve by Image")
    button1 = Button(window2_main,text ='Open Image',command = lambda: loadImg(window2_main), height=2, width=20)
    button2 = Button(window2_main,text ='Solve',command = lambda: solve("test.jpg", window2_main), height=2, width=20)
    button3 = Button(window2_main,text ='Close',command = lambda: window1(window2_main), height=2, width=20)
    button1.grid(row=0, column=0, padx=(20, 20), pady=(20, 20))
    button2.grid(row=0, column=1, padx=(20, 20), pady=(20, 20))
    button3.grid(row=0, column=2, padx=(20, 20), pady=(20, 20))
    window2_main.mainloop()

window1(root)
root.mainloop()
