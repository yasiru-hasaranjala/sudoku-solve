import sys
from tkinter import * #or Tkinter if you're on Python2.7
from PIL import Image, ImageTk
import tkinter.filedialog as tkFileDialog
import sudoku
from sudoku import *
import ntpath
import os
imgPath = ""

def button1():
    path=tkFileDialog.askopenfilename(filetypes=[("Image File",'.jpg')])
    im = Image.open(path)
    im.save("test.jpg")
    imgPath = "test.jpg"
    width, height = im.size
    im = im.resize((int(width*450/height) ,450), Image.ANTIALIAS)
    tkimage = ImageTk.PhotoImage(im)
    myvar=Label(root,image = tkimage)
    myvar.image = tkimage
    myvar.grid(row=1,column=0, columnspan=3, sticky="nsew")

def solve(path):
    solveByImage(path)
    im = Image.open('solved.jpg')
    width, height = im.size
    im = im.resize((int(width*450/height) ,450), Image.ANTIALIAS)
    tkimage = ImageTk.PhotoImage(im)
    myvar=Label(root,image = tkimage)
    myvar.image = tkimage
    myvar.grid(row=1,column=0, columnspan=3, sticky="nsew")


root = Tk()
button1 = Button(root,text ='Open Image',command = button1, height=2, width=20)
button2 = Button(root,text ='Solve',command = lambda: solve("test.jpg"), height=2, width=20)
button3 = Button(root,text ='Close',command = root.destroy, height=2, width=20)
button1.grid(row=0, column=0, padx=(20, 20), pady=(20, 20))
button2.grid(row=0, column=1, padx=(20, 20), pady=(20, 20))
button3.grid(row=0, column=2, padx=(20, 20), pady=(20, 20))
root.mainloop()
