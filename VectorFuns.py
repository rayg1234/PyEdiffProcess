# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:34:00 2014

This lib contains a set of functions for general operations on vectors and matrices

@author: Ray Gao, 2014
"""

import numpy as PY
from numpy import linalg as LA
from sympy import Matrix
from UtilFuns import *

"""
find_basis_set uses row echelon form to find a set of vectors that span the space

[Inputs]
vecs: the input vectors

[Returns]
the basis set of vectors
"""
def find_basis_set(vecs,**kwargs):
    A = Matrix(vecs)
    basiscols = A.rref()[1]
    
    return PY.double(vecs[:,PY.array(basiscols)])

"""
get_reci_vecs2 returns the reciprocal lattice vectors in 2D for a given set of "real" lattice vectors.

[Inputs]
vec1,vec2: the "real" lattice vectors
fflen: the fourier transform length (ie: the linear dimension of the original image from which
    the fourier transform is computed (assuming a square)). this is the proper scaling factor 
    to convert from fourier space to real space

[Returns]
two reciprocal lattice vectors
"""
def get_reci_vecs2(vec1,vec2,fflen):
    newv1 = fflen * rotate_vec(vec2,PY.pi/2)/(PY.dot(vec1,rotate_vec(vec2,PY.pi/2)))
    newv2 = fflen * rotate_vec(vec1,-PY.pi/2)/(PY.dot(vec2,rotate_vec(vec1,-PY.pi/2)))
    return newv1,newv2

"""
angle_btw_vecs computes the angle between two vectors

[Inputs]
vec1,vec2: the two vectors

[Returns]
the angle
"""
def angle_btw_vecs(vec1,vec2):
    return PY.arccos(PY.dot(vec1,vec2)/(LA.norm(vec1)*LA.norm(vec2)))

"""
rotate_vec applies a 2D rotation matrix op to the given vector

[Inputs]
vec: the vector to be rotated
theta: the angle in radians

[Returns]
the rotated vector
"""
def rotate_vec(vec,theta):
    R = PY.mat([[PY.cos(theta),-PY.sin(theta)],[PY.sin(theta),PY.cos(theta)]])
    newm = PY.dot(R,vec) 
    return PY.array([newm[0,0],newm[0,1]])

"""
cart_to_polar converts the given vector in cartesian coords to polar form

[Inputs]
vec: the vector to be converted

[Returns]
the polar form of the vector [magnitude,angle]
"""
def cart_to_polar(vec):
    if(vec.shape[0] is not 2):
        print 'shape incorrect for conversion... exiting...'
        return
    vec2 = PY.zeros(vec.shape)
    vec2[0] = PY.sqrt(vec[0]**2 + vec[1]**2)
    vec2[1] = PY.arctan(vec[1]/vec[0])
    return vec2

"""
polar_to_cart converts the given vector in polar form to cartesian form

[Inputs]
vec: the vector to be converted

[Returns]
the cartesian form of the vector [x,y]
"""
def polar_to_cart(vec):
    if(vec.shape[0] is not 2):
        print 'shape incorrect for conversion... exiting...'
        return
    vec2 = PY.zeros(vec.shape)
    vec2[1] = vec[0]*PY.sin(vec[1])
    vec2[0] = vec[0]*PY.cos(vec[1])
    return vec2

"""
*** depricated ***
a poorly implemented method for vector filling
use fill_space_with_basis2D_2 instead
"""
def fill_space_with_basis2D(Arrimg,vec1,vec2,cxy):
    #calculate ray for vec1
    l1 = LA.norm(vec1)
    l2 = LA.norm(vec2)
    theshape = PY.asanyarray(Arrimg.shape)
    #first slope 
    m1 = vec1[0]/vec1[1]
    #x = 0 (second coord)
    b1a = cxy[0] - m1*cxy[1]
    b1b = m1*Arrimg.shape[1]+b1a
    #y = 0 (first coord)
    c1a = -b1a/m1
    c1b = (Arrimg.shape[0] -b1a)/m1
    #print (b1a)
    #print (b1b)
    #print (c1a)
    #print (c1b)
    if(b1a >= 0):
        lowind1 = -PY.int64(LA.norm(cxy - PY.array([b1a,0]))/l1)
    else:
        lowind1 = -PY.int64(LA.norm(cxy - PY.array([0,c1a]))/l1)
    if(b1b < theshape[1]):
        highind1 = PY.int64( LA.norm(theshape - PY.array([b1b,0])) /l1)
    else:
        highind1 = PY.int64( LA.norm(theshape - PY.array([0,c1b])) /l1)
        
    #second slope
    m2 = vec2[0]/vec2[1]
    #x = 0 (second coord)
    b2a = cxy[0] - m2*cxy[1]
    b2b = m2*Arrimg.shape[1]+b2a
    #y = 0 (first coord)
    c2a = -b2a/m2
    c2b = (Arrimg.shape[0] -b2a)/m2
    #print (b2a)
    #print (b2b)
    #print (c2a)
    #print (c2b)
    if(b2a >= 0):
        lowind2 = -PY.int64(LA.norm(cxy - PY.array([b2a,0]))/l2)
    else:
        lowind2 = -PY.int64(LA.norm(cxy - PY.array([0,c2a]))/l2)
    if(b2b < theshape[1]):
        highind2 = PY.int64( LA.norm(theshape - PY.array([b2b,0])) /l2)
    else:
        highind2 = PY.int64( LA.norm(theshape - PY.array([0,c2b]))/l2)
    
    numvecs = (highind1*2 - lowind1*2)*(highind2*2 - lowind2*2)
    thevecs = PY.zeros([numvecs,2])
    count = 0
    for i in range(lowind1*2,highind1*2):
        for j in range(lowind2*2,highind2*2):
            curvec  = i*vec1 + j*vec2 + cxy
            if(curvec[0]>0 and curvec[1]>0 and curvec[0]<theshape[0] and curvec[1]<theshape[1]):
                thevecs[count,:] = curvec                
                count += 1
    return thevecs[0:count,:]


"""
fill_space_with_basis2D_2 (Vectorized) completely fills a 2D space with points given two input vectors
by computing all linear combinations of the input vectors that fits into the space

[Input]
Arrimg: the vectors will be filled up to the dimensions of this image
vec1,vec2: the two basis vectors
cxy: the center to start filling

[Return]
a n x 2 array of vectors that represents the set of linear combinations of the input vectors that fills the space
"""
def fill_space_with_basis2D_2(Arrimg,vec1,vec2,cxy):
    minc = -20
    maxc = 20
    tot = (maxc-minc)*(maxc-minc)
    x = PY.arange(minc,maxc)
    y = PY.arange(minc,maxc)
    xx,yy = PY.meshgrid(x,y)
    lcombx = xx*vec1[0] + yy*vec2[0] + cxy[0]
    lcomby = xx*vec1[1] + yy*vec2[1] + cxy[1]
    lcombx = lcombx.reshape((tot,1))
    lcomby = lcomby.reshape((tot,1))
    delete1 = PY.nonzero((lcombx>=Arrimg.shape[0]) | (lcombx<0))[0]
    lcombx = PY.delete(lcombx,delete1)
    lcomby = PY.delete(lcomby,delete1)
    delete2 = PY.nonzero((lcomby>=Arrimg.shape[1]) | (lcomby<0))[0]
    lcombx = PY.delete(lcombx,delete2)
    lcomby = PY.delete(lcomby,delete2)
    
    return PY.vstack([lcombx,lcomby]).T

    