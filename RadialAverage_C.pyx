# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:07:49 2014

This library cotanins the set of functions related to taking the radial average
of diffraction patterns. These functions are implemented in Cython because
they are difficult to completely vectorize and benefit from the significantly
from the for loop speedup.

@author: Ray Gao, 2014
"""

import numpy as np
cimport numpy as np
DTYPE = np.double
DTYPE2 = np.int
ctypedef np.double_t DTYPE_t
ctypedef np.int_t DTYPE2_t
#from math import sqrt,round


"""
RadialAverage_fastsq computes the radial averages as a function of the squares of the distances
from the center. This benefits from a slight speedup by avoiding the sqrt function.

[Inputs]
Arrimg: The original image containing the diffraction pattern
cx,cy: the center to take the radial average from

[Returns]
An array containing the radial averages as a function of the square of the distances from the center

"""
def RadialAverage_fastsq(np.ndarray[DTYPE_t, ndim=2] Arrimg,int cx,int cy):
    assert Arrimg.dtype == DTYPE

    cdef int maxR = np.max([Arrimg.shape[0]-cx,Arrimg.shape[1]-cy])**2
    cdef np.ndarray[DTYPE_t, ndim=2] R = np.zeros([maxR,2],dtype=DTYPE)
    cdef int xsize = Arrimg.shape[0]
    cdef int ysize = Arrimg.shape[1]
    cdef int i,j,d
    cdef DTYPE_t s
    
    for i in range(xsize):
        for j in range(ysize):
            #s += Arrimg[i,j]            
            d = (i-cx)**2 + (j-cy)**2          
            if(d<maxR):
                R[d,0] += Arrimg[i,j] 
                R[d,1] += 1
                
    cdef np.ndarray nzelems = np.nonzero(R[:,1])[0]
    R[nzelems,0]=R[nzelems,0]/R[nzelems,1]
    
    return R[:,0]

"""
RadialAverage_fast computes the radial averages as a function of the distances from the center

[Inputs]
Arrimg: The original image containing the diffraction pattern
cx,cy: the center to take the radial average from

[Returns]
An array containing the radial averages as a function of the distance from the center

"""
def RadialAverage_fast(np.ndarray[DTYPE_t, ndim=2] Arrimg,int cx,int cy):
    assert Arrimg.dtype == DTYPE

    cdef int maxX = np.max([Arrimg.shape[0]-cx,cx])
    cdef int maxY = np.max([Arrimg.shape[1]-cy,cy])
    cdef int maxR = np.round(np.sqrt(maxX**2 + maxY**2))
    cdef np.ndarray[DTYPE_t, ndim=2] R = np.zeros([maxR,2],dtype=DTYPE)
    cdef int xsize = Arrimg.shape[0]
    cdef int ysize = Arrimg.shape[1]
    cdef int i,j
    cdef long n
    
    cdef np.ndarray x = np.arange(0,xsize)
    cdef np.ndarray y = np.arange(0,ysize)
    cdef np.ndarray xx
    cdef np.ndarray yy
    xx,yy = np.meshgrid(x,y)
    cdef np.ndarray[DTYPE_t, ndim=2] d = np.round(np.sqrt((xx-cy)**2 + (yy-cx)**2)).T
    
#    for i in range(maxR):
#        R[i] = sum(Arrimg[np.where(d==i)])
    for i in range(xsize):
        for j in range(ysize):
            n = <long>d[i,j]
            if(n<maxR):
                R[n,0] += Arrimg[i,j] 
                R[n,1] += 1
    
    cdef np.ndarray nzelems = np.nonzero(R[:,1])[0]
    R[nzelems,0]=R[nzelems,0]/R[nzelems,1]
    
    return R[:,0]

"""
RadialAverage_gzero computes the radial averages as a function of the distances from the center.
This is similar to RadialAverage_fast except only pixel values that are greater than 0 contribute to the sums

[Inputs]
Arrimg: The original image containing the diffraction pattern
cx,cy: the center to take the radial average from

[Returns]
An array containing the radial averages as a function of the distance from the center

"""
def RadialAverage_gzero(np.ndarray[DTYPE_t, ndim=2] Arrimg,int cx,int cy):
    assert Arrimg.dtype == DTYPE

    cdef int maxX = np.max([Arrimg.shape[0]-cx,cx])
    cdef int maxY = np.max([Arrimg.shape[1]-cy,cy])
    cdef int maxR = np.round(np.sqrt(maxX**2 + maxY**2))
    #cdef int maxR = np.max([Arrimg.shape[0]-cx,Arrimg.shape[1]-cy])
    cdef np.ndarray[DTYPE_t, ndim=2] R = np.zeros([maxR,2],dtype=DTYPE)
    cdef int xsize = Arrimg.shape[0]
    cdef int ysize = Arrimg.shape[1]
    cdef int i,j
    cdef long n
    
    cdef np.ndarray x = np.arange(0,xsize)
    cdef np.ndarray y = np.arange(0,ysize)
    cdef np.ndarray xx
    cdef np.ndarray yy
    cdef DTYPE_t val
    xx,yy = np.meshgrid(x,y)
    cdef np.ndarray[DTYPE_t, ndim=2] d = np.round(np.sqrt((xx-cy)**2 + (yy-cx)**2)).T
    
#    for i in range(maxR):
#        R[i] = sum(Arrimg[np.where(d==i)])
    for i in range(xsize):
        for j in range(ysize):
            n = <long>d[i,j]
            val = Arrimg[i,j] 
            if(n<maxR and val>0):
                R[n,0] += val
                R[n,1] += 1
    
    cdef np.ndarray nzelems = np.nonzero(R[:,1])[0]
    R[nzelems,0]=R[nzelems,0]/R[nzelems,1]
    
    return R[:,0]

"""
RadialAverage_toImg converts a radial average function into a 2D image.

[Inputs]
R: the radial average vector
xsize,ysize: the sizes of the image
cx,cy: the center to take the radial average from

[Returns]
the image representation of the radial average

"""
def RadialAverage_toImg(np.ndarray[DTYPE_t, ndim=1] R,int xsize,int ysize,int cx,int cy):
    assert R.dtype == DTYPE
    
    cdef np.ndarray[DTYPE_t, ndim=2] theimg = np.zeros([xsize,ysize],dtype=DTYPE)
    
    cdef long Rsize = R.shape[0]
    cdef np.ndarray x = np.arange(0,xsize)
    cdef np.ndarray y = np.arange(0,ysize)
    cdef np.ndarray xx
    cdef np.ndarray yy
    cdef int i,j    
    cdef long n
    xx,yy = np.meshgrid(x,y)
    cdef np.ndarray[DTYPE_t, ndim=2] d = np.round(np.sqrt((xx-cy)**2 + (yy-cx)**2)).T    
    
    for i in range(xsize):
        for j in range(ysize):
            n = <long>d[i,j]
            if(n<Rsize):
                theimg[i,j] = R[n]
                
    return theimg