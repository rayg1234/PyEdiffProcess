# -*- coding: utf-8 -*-
"""
Created on Wed Feb 05 13:15:39 2014

This library contains a set of functions for cleaning and analyzing raw diffraction patterns

@author: Ray Gao, 2014
"""
import Image as IMG
import numpy as PY
import matplotlib.pyplot as PLT
import scipy.ndimage as NDIMG
from skimage import filter
from RadialAverage_C import *
from imshow_mod import imshow_z
from copy import copy

"""
TifftoArray converts a Tiff image opened with PIL library to a numpy array 

[Inputs]
img: the TIFF image loaded using PIL

[Returns]
ImgArray: the numpy array containing the image

"""
def TifftoArray(img):

    ImgArray = PY.zeros((img.size[0],img.size[1]))
    imgd = PY.asarray(img.getdata());
    for i in range(0,img.size[1]):
        ImgArray[:,i] = imgd[(i*img.size[0]):((i+1)*img.size[0])]
    return ImgArray.T;

"""
Uses the canny method to build a glaremask (to eliminated saturated pixels)

[Inputs]
Arrimg: a numpy 2D array containing the original image

[Returns]
a 2D array with same input dimensions containing the glaremask

"""
def Glaremask(Arrimg):
    #edge1 = filter.canny(Arrimg,sigma=1.8,high_threshold=0,low_threshold=200)
    edge1 = filter.canny(Arrimg,sigma=3,low_threshold=2000,high_threshold=25000);
    filled = NDIMG.binary_fill_holes(edge1);
    return 1-filled;

"""
Uses a sobel filter and a threshold to build a glare mask to elminate the saturated
pixels. This tends to work better than Glaremask (using the canny detector) as
it will return a less precise region but performs better at connecting edges.

[Inputs]
Arrimg: a numpy 2D array containing the original image
optional args:
highthresh: the high cutoff threshold

[Returns]
a 2D array with same input dimensions containing the glaremask

"""
def GlaremaskSobel(Arrimg,**kwargs):
    highthresh = kwargs.get('highthresh', None)
    if highthresh is None:
        highthresh = 2000    
    sob = filter.sobel(Arrimg)
    sob[sob<highthresh] = 0
    sob[sob>0] = 1
    filled = NDIMG.binary_fill_holes(sob);
    return 1-filled

"""
Circularmask creates a filled circle shaped mask function

[Inputs]
Arrimg: a numpy 2D array containing the original image
cxy: vector describing the center of where the mask should be
r: radius of circle

[Returns]
a 2D array with same input dimensions containing the circlemask

"""
def Circularmask(Arrimg,cxy,r):
    x = PY.arange(0,Arrimg.shape[0])
    y = PY.arange(0,Arrimg.shape[1])
    xx,yy = PY.meshgrid(x,y)
    d = PY.round(PY.sqrt((xx-cxy[1])**2 + (yy-cxy[0])**2)).T
    A = PY.ones(Arrimg.shape)
    A[d<r] = 0
    return A

"""
DiffCenter locates the center of a diffraction pattern image by intergrating over
all rows and columns and finding the max of the auto-correlation.

[Inputs]
Arrimg: a numpy 2D array containing the original image

[Returns]
[xcenter,ycenter]
"""
def DiffCenter(Arrimg):
    #g = Glaremask(Arrimg);
    #newimg = Arrimg*g;
    sx = PY.sum(Arrimg,axis=0);
    sy = PY.sum(Arrimg,axis=1);
    #PLT.plot(sx)
    #PLT.plot(sy)
    sxc= PY.correlate(sx,PY.flipud(sx),mode='full');
    syc= PY.correlate(sy,PY.flipud(sy),mode='full');
    xcent = PY.nonzero(sxc==sxc.max())[0][0]/2;
    ycent = PY.nonzero(syc==syc.max())[0][0]/2;
    return [xcent,ycent];

"""
DiffCenter_Simple Locates the center of a diffraction pattern image by intergrating over
all rows and columns and finding the max of these values. This skips the correlation
step from DiffCenter.

[Inputs]
Arrimg: a numpy 2D array containing the original image

[Returns]
[xcenter,ycenter]
"""
def DiffCenter_Simple(Arrimg):
    sx = PY.sum(Arrimg,axis=0);
    sy = PY.sum(Arrimg,axis=1);
    xcent = PY.nonzero(sx==sx.max())[0][0];
    ycent = PY.nonzero(sy==sy.max())[0][0];
    return [xcent,ycent];    

"""
DiffBGExtract (Vectorized) removes the radial background from a diffraction image
a) taking the radial average  
b) remove the radial average from the original image to generate the diffraction
pattern with all the peaks removed
c) create a mask where the peaks by using the previous image
d) take a second radial average with the mask function to extract the real background
e) remove the real background from the original image

[Inputs]
Arrimg: a numpy 2D array containing the original image

Optional arguments
cxy: the center where the radial average should be taking, otherwise the function
attempts to autodetect the center

[Returns]
The diffraction pattern with the radial background removed
"""    
def DiffBGExtract(Arrimg,**kwargs):
    cxy = kwargs.get('cxy', None)
    if cxy is None:
        #print 'Generating center...'
        g = GlaremaskSobel(Arrimg)
        cxy = DiffCenter(Arrimg*g)
        #print (cxy)
#    cm = Circularmask(Arrimg,cxy,70)
    newimg = Arrimg
    R = RadialAverage_fast(newimg,cxy[0],cxy[1])
    Rimg = RadialAverage_toImg(R,Arrimg.shape[0],Arrimg.shape[1],cxy[0],cxy[1])
    bgsub = newimg - Rimg
    Arrimg_nopeak = copy(newimg)
    Arrimg_nopeak[PY.nonzero(bgsub>0)] = 0
    #imshow_z(Arrimg_nopeak,vmax=1000)
    R2 = RadialAverage_gzero(Arrimg_nopeak,cxy[0],cxy[1])
    Rimg2 = RadialAverage_toImg(R2,Arrimg.shape[0],Arrimg.shape[1],cxy[0],cxy[1])
    return newimg - Rimg2

"""
DiffReshape reshapes the given diffraction image with new dimensions by either cropping
or padding.
***Need to be fixed:
currently only works with newshape with dimensions that are even integers

[Input]
Arrimg: a numpy 2D array containing the original image
cxy: the center to crop from
newshpae: the dimensions of the new image.
    if a new dimension is bigger than the original image, the new image is padded with zeros
    if a new dimension is smaller than the original image, the new image is a cropped version of the old image

[Return]
The image with the new shape 

"""    
    #currently does not work for newshape with odd dimensions
def DiffReshape(Arrimg,cxy,newshape):
    oldshape = Arrimg.shape
    newimg = PY.zeros(newshape)
    if cxy[1]<=round(newshape[0]/2):
        xleftold = 0
        xleftnew = round(newshape[0]/2) - cxy[1]
    else:
        xleftold = cxy[1] - round(newshape[0]/2)
        xleftnew = 0
    if (oldshape[0] - cxy[1])<=round(newshape[0]/2):
        xrightold = oldshape[0]
        xrightnew = round(newshape[0]/2) + (oldshape[0] - cxy[1])
    else:
        xrightold = cxy[1] + round(newshape[0]/2)
        xrightnew = newshape[0]
        
    if cxy[0]<=round(newshape[1]/2):
        yleftold = 0
        yleftnew = round(newshape[1]/2) - cxy[0]
    else:
        yleftold = cxy[0] - round(newshape[1]/2)
        yleftnew = 0
    if (oldshape[1] - cxy[0])<=round(newshape[1]/2):
        yrightold = oldshape[1]
        yrightnew = round(newshape[1]/2) + (oldshape[1] - cxy[0])
    else:
        yrightold = cxy[0] + round(newshape[1]/2)
        yrightnew = newshape[1]
    
    newimg[xleftnew:xrightnew,yleftnew:yrightnew] = Arrimg[xleftold:xrightold,yleftold:yrightold]
    return newimg

    
"""
DiffCircleCrop (Vectorized) crops a circular region in the original image

[Input]
Arrimg: a numpy 2D array containing the original image
cxy: the center to crop from
r: the radius of the crop

[Return]
a cropped version of the original image (but still represented by a rectangular array). The outside of the circle is padded with zeros.
"""    
def DiffCircleCrop(Arrimg,cxy,r):
    newimg = DiffReshape(Arrimg,cxy,[2*r,2*r])
    x = PY.arange(0,2*r)
    y = PY.arange(0,2*r)
    xx,yy = PY.meshgrid(x,y)
    d = PY.round(PY.sqrt((xx-r)**2 + (yy-r)**2)).T
    newimg[d>r] = 0
    return newimg


"""
DiffNCircleMask (Vectorized) creates mask that contains an array of circles with center positions 
given by cxy and radius r. This function uses a sequence of convolution operations
to avoid looping through the list of circles. However, performance is comparable
after testing as convolutions are expensive ops. The masks also deviate from true
circular shapes depending on the convolution kernel used.

[Input]
Arrimg: a numpy 2D array containing the original image
cxy: n x 2 array of centers to create the mask from
r: the radius of the circles

[Return]
The mask
"""    
def DiffNCircleMask(Arrimg,cxy,r):
    mask = PY.zeros(Arrimg.shape)
    mask[list(cxy[:,0]),list(cxy[:,1])] = 1
    #k =  PY.array([[1/16,2/16,1/16],[2/16,4/16,2/16],[1/16,2/16,1/16]])    
    k =  PY.array([[1,2,1],[2,4,2],[1,2,1]])    
    #k = PY.array([1,4,6,4,1])
    #k = PY.outer(k,k)
    #k = PY.array([[1,1,1],[1,2,1],[1,1,1]])
    for i in range(0,int(round((r/2)))):
        mask = NDIMG.convolve(mask,k)
    #thresh = r*1
    #mask[mask<thresh] = 0
    mask[mask>0] = 1
    return mask

"""
Incomplete
"""
def DiffPointMask(Arrimg,cxy):
    mask = PY.zeros(Arrimg.shape)
    mask[list(cxy[:,0]),list(cxy[:,1])] = 1
    return mask

"""
DiffAutoCorr takes advantage of the convolution theorem to compute the autocorrelation
of an image quickly. Autocorrelation = FFT^-1 [FFT(img) x FFT*(img)]. For large images,
this method is significantly faster than brute force autocorrelation.

[Input]
Arrimg: the original image
s: the newshape to transform the image before computing the autocorrelation. since
this method uses ffts, it is most efficient when the dimensions are multiples of 2's

Optional arguments
cxy: the center to transform the newshape
mask: a mask to apply before computing the convolution
masksize: auto-generate a circlularmask with this radius

[Return]
The auto-correlation of the original image

"""
def DiffAutoCorr(Arrimg,s,**kwargs):
    cxy = kwargs.get('cxy', None)
    mask = kwargs.get('mask', None)
    masksize = kwargs.get('masksize', None)
    if cxy is None:
        print 'Generating center...'
        g = Glaremask(Arrimg)
        cxy = DiffCenter(Arrimg*g)
        print (cxy)
    if mask is None:
        mask = PY.ones(Arrimg.shape)
    if masksize is not None:
        mask = Circularmask(Arrimg,cxy,masksize)  
        print 'Generating new mask...'
    newimg = DiffReshape(Arrimg*mask,cxy,s)
    Arrfft = PY.fft.fftshift(PY.fft.fft2(newimg,s)) 
    #res = PY.fft.ifft2(Arrfft)
    res = PY.abs(PY.fft.fftshift(PY.fft.ifft2(Arrfft*PY.conjugate(Arrfft)))/PY.sum(abs(newimg)))
    return res

"""
Centroid (Vectorized) computes the centroid (ie: weighted mean) of the image

[Input]
Arrimg: the original image

Optional arguments
mask: a mask to apply before computing the centroid

[Return]
The centroid [cx,cy]

"""    
def Centroid(Arrimg,**kwargs):
    mask = kwargs.get('mask',None)
    if mask is None:
        mask = PY.ones(Arrimg.shape)
        #print (mask.shape)
    x = PY.arange(0,Arrimg.shape[0])
    y = PY.arange(0,Arrimg.shape[1])
    xx,yy = PY.meshgrid(x,y)
    cx = PY.sum(xx.T*Arrimg*mask)/PY.sum(Arrimg*mask)
    cy = PY.sum(yy.T*Arrimg*mask)/PY.sum(Arrimg*mask)
    return PY.array([cx,cy])

"""
Max2D computes the location and value of the maxima in the image

[Input]
Arrimg: the original image

[Return]
An array containing x,y and the value [x,y,val]

"""    
def Max2D(Arrimg):
    ind = PY.argmax(Arrimg)
    x = PY.ceil(ind/Arrimg.shape[0])
    y = ind%Arrimg.shape[0]
    val = Arrimg[x,y]
    return [x,y,val]

"""
PeakFilteredFFT creates a "clean" fourier transform of the image. This is used
to clearly identify the fundamental frequencies in a diffraction image. The fft
is cleaned up by computing the laplacing function identify the locations of the
peaks in the fft and then thresholding to recover only the strongest peaks.

[Input]
Arrimg: the original image

Optional arguments:
sigma: this is used as the thresholding parameter. Only peaks with laplacian values
greater than sigma times the standard deviation (sigma times above the noise) are retained.
The greater sigma, the more stringent the thresholding.

[Return]
A filtered FFT to clearly identify the fundamental frequencies in the image.

"""        
def PeakFilteredFFT(Arrimg,cxy,**kwargs):
    sigma = kwargs.get('sigma',None)
    if sigma is None:
        sigma = 10
    #autoc = DiffAutoCorr(filt,s,cxy=cxy)
    #take laplacian    
    thefft = PY.fft.fftshift(PY.fft.fft2(Arrimg))
    
    lap = NDIMG.filters.gaussian_laplace(Arrimg,3)    
    #take fft of laplacian 
    lapfft = PY.fft.fftshift(PY.fft.fft2(lap))
    
    lapfftabs = abs(lapfft)
    #take laplacian again
    lapfftlap = NDIMG.filters.gaussian_laplace(lapfftabs,1)
    
    #remove noise from fft
    lapfftlapthresh = -PY.std(lapfftlap)*sigma
    #print (lapfftlapthresh)
    #lapfftbin = copy(lapfftlap)
    #lapfftbin[lapfftlap>lapfftlapthresh]=0
    #lapfftbin[lapfftlap<lapfftlapthresh]=1
    
    #lapfftfilt = copy(lapfft)
    #lapfftfilt[lapfftlap>lapfftlapthresh] = 0
    thefft[lapfftlap>lapfftlapthresh] = 0
    #smallcm = Circularmask(thefft,PY.asanyarray(Arrimg.shape)/2,5)
    #imshow_z(abs(thefft)*smallcm)
    
    #take ifft
    #lap2 = abs(PY.fft.ifft2(thefft))
    #imshow_z(lap2)
    return thefft

"""
FindPeakPos uses the laplacian/thresholding to help find the peak positions in an image.
Note using this function directly on a diffraction pattern is not an effective solution
unless all the peaks are extremely intense and sharp.

Instead use the PeakFilteredFFT to find the fundamental frequencies first, use this
function to get the positions of the fundamental frequencies and then find the reciprocal 
lattice vectors that way.

[Input]
Arrimg: the original image

Optional arguments:
thresh: this is used as the thresholding parameter. Only peaks with laplacian values
greater than sigma times the standard deviation (sigma times above the noise) are retained.
The greater sigma, the more stringent the thresholding.

[Return]
A n x 5 vector containing the positions of the all peaks
the positions are given both using the centroid function and the max2D function
the first 3 cols represent the values given max2D [x,y,val]
the last 2 cols represent the values given by centroid [cx,cy]
"""  
def FindPeakPos(Arrimg,**kwargs):
    thresh = kwargs.get('thresh',None)
    if thresh is None:
        thresh = 10
    lap = NDIMG.filters.gaussian_laplace(Arrimg,3)
    lapthresh = -PY.std(lap)*thresh
    lapbin = copy(lap)
    lapbin[lap>lapthresh]=0
    lapbin[lap<lapthresh]=1
    laplabels,numfeatures = NDIMG.measurements.label(lapbin);
    cms = NDIMG.measurements.center_of_mass(lapbin,laplabels,PY.arange(1,numfeatures))
    cms = PY.asanyarray(cms)
    pos = PY.zeros([numfeatures,5])
    for i in range(0,numfeatures):
        pksingle = PY.zeros(lap.shape)
        pksingle[(laplabels == i+1)] =Arrimg[(laplabels == i+1)]
        pos[i,0:3] = Max2D(pksingle)
        pos[i,3:5] = Centroid(pksingle)
    
    return pos

"""
draw_circles_along_vec draws a line of circles along the direction of a given vector
for visualization. 

[Input]
thefig: the figure handle
vec: the direction vector
cxy: the center to start drawing
lims: how far to extend the circles

Optional arguments:
color: the color of the circles

[Return]
none
"""      
def draw_circles_along_vec(thefig,vec,cxy,lims,**kwargs):
    thecolor = kwargs.get('color',None)
    if thecolor is None:
        thecolor = 'r'
    cxy = PY.array(cxy)
    curvec = copy(vec+cxy)
    i = 1;
    while (curvec[1]<lims[0] and curvec[0]<lims[1] and curvec[1]>0 and curvec[0]>0):
        circle1=PLT.Circle((curvec[1],curvec[0]),10,color=thecolor,fill=False,lw=2)
        thefig.gca().add_artist(circle1)
        curvec = vec*i + (cxy)
        i += 1
    
    thefig.canvas.draw()

"""
draw_circles_at_vec draws circles at the positions given by a set of vectors
for visualization

[Input]
thefig: the figure handle
vec1: n x 2 array, the list of vectors to the draw the circles at

Optional arguments:
color: the color of the circles
size: the size of the circles

[Return]
none
"""     
def draw_circles_at_vec(thefig,vec1,**kwargs):
    thecolor = kwargs.get('color',None)
    thesize = kwargs.get('size',None)
    if thecolor is None:
        thecolor = 'r'
    if thesize is None:
        thesize = 10
    for i in range(vec1.shape[0]):
        circle1=PLT.Circle((vec1[i,1],vec1[i,0]),thesize,color=thecolor,fill=False,lw=1)
        thefig.gca().add_artist(circle1)
    
    thefig.canvas.draw()

"""
DiffBGStats extracts the background mean and standard deviation of the background from a diffractionimage

[Input]
Arrimg: the original image
cxy: the center of the diffraction pattern

[Return]
mean, standard deviation
"""     
def DiffBgStats(Arrimg,cxy):
    R = RadialAverage_fast(Arrimg,cxy[0],cxy[1])
    Rimg = RadialAverage_toImg(R,Arrimg.shape[0],Arrimg.shape[1],cxy[0],cxy[1])
    bgsub = Arrimg - Rimg
    Arrimg_nopeak = copy(Arrimg)
    Arrimg_nopeak[PY.nonzero(bgsub>0)] = 0
    mean1 = PY.mean(Arrimg_nopeak)
    std1 = PY.std(Arrimg_nopeak)
    return mean1,std1

"""
DiffPeakSums extracts the integral of the peak intensities given the pre-identified peak locations.
The peak locations are corrected slightly by using the centroid to shift the peak locations
towards maxima and collect the sums in the shifted location.

[Input]
Arrimg: the original image
thevecs: the locations of the peaks

Optional Argument:
rad: the radius of the circle to integrate over for each peak

[Return]
cents1: n x 2 array containing the positions where the sums where extracted from
sums: n array containing the integral of the peak intensities
"""     
def DiffPeakSums(Arrimg,thevecs,**kwargs):
    rad = kwargs.get('rad',None)
    if rad is None:
        rad = 30
    thevecs_round = PY.round(thevecs)
    meanbg, stdbg = DiffBgStats(Arrimg,[512,512])
    lap = NDIMG.filters.gaussian_laplace(Arrimg,4)
    #laps1 = PY.zeros(thevecs.shape[0])
    mean1 = PY.zeros(thevecs.shape[0])
    cents1 = PY.zeros(thevecs.shape)
    sums = PY.zeros(thevecs.shape[0])
    #filtvecs1 = PY.zeros(thevecs.shape)
    count = 0
    for i in range(thevecs.shape[0]):
        thecrop = DiffCircleCrop(Arrimg,thevecs_round[i,::-1],rad)
        thecroplap = DiffCircleCrop(lap,thevecs_round[i,::-1],rad)
        mean1[i] = PY.sum(thecrop)/(PY.pi* rad**2)
        lapval = PY.sum(thecroplap)/(PY.pi* rad**2)
        if(mean1[i]>(2*stdbg - meanbg) and lapval<-0.01):
            #filtvecs1[count,:] = thevecs[i,:]
            cents1[i,:] = Centroid(thecrop) + thevecs[i,:] - PY.array([rad,rad])
            newcrop = DiffCircleCrop(Arrimg,cents1[i,::-1],rad)
            sums[i] = PY.sum(newcrop)
            count = count +1
        else:
            cents1[i,:] = thevecs[i,:]
            sums[i] = mean1[i] * (PY.pi * rad**2)
            
    return cents1,sums
    