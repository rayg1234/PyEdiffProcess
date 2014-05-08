# -*- coding: utf-8 -*-
"""
Created on Wed Feb 05 15:06:13 2014

@author: Ray
"""
import numpy as np
import matplotlib.pyplot as plt
import time
    

def clearall():
    
    for uniquevar in [var for var in globals().copy() if var[0] != "_" and var != 'clearall']:
        del globals()[uniquevar]
#Homemade version of matlab tic and toc functions
def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"