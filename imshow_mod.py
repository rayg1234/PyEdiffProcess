# -*- coding: utf-8 -*-
"""
Created on Fri Feb 07 15:03:41 2014

@author: Ray
"""
import numpy as py
import matplotlib.pyplot as plt

class imshow_helper:
    
    def __init__(self,ax,img):
        self.ax = ax
        self.img = img
        self.ax.format_coord = self.format_coord
        self.numrows, self.numcols = self.img.shape
        
    def format_coord(self,x,y):
        col = int(x+0.5)
        row = int(y+0.5)
        if col>=0 and col<self.numcols and row>=0 and row<self.numrows:
            z = self.img[row,col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f'%(x, y)

def imshow_z(ArrImg,*args,**kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)        
    ax.imshow(ArrImg,*args,**kwargs)    
    imshow_helper(ax,ArrImg)
    plt.show()
    return fig