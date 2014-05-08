# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:10:06 2014

@author: Ray
"""

import Image as IMG
import numpy as np
import scipy.ndimage as NDIMG
import matplotlib.pyplot as PLT
import ImageUtils as IU
from UtilFuns import tic,toc
from imshow_mod import imshow_z
import RadialAverage_C as RC
import VectorFuns as VF
import CrystalAlg_C as CALG
import scipy.optimize as SCO
import VisualStructureFactors as VS
from mpl_toolkits.mplot3d import Axes3D

img = IMG.open('D:/Ediff/PythonWorks/EDO_LTavg16.tif')
ArrImg = IU.TifftoArray(img)
imshow_z(ArrImg,vmin=0,vmax=1000)

##########################################################################
#clean data
##########################################################################
#remove saturation
g = IU.GlaremaskSobel(ArrImg,highthresh=700)
#imshow_z(g*ArrImg)
#cxy = [450,450]

#remove background
cxy = IU.DiffCenter(ArrImg*g)
cm = IU.Circularmask(ArrImg,cxy,50)
R = RC.RadialAverage_fast(ArrImg*g,cxy[0],cxy[1])
#PLT.figure()
#PLT.plot(R)
bgsub = IU.DiffBGExtract(ArrImg,cxy=cxy)
#imshow_z(bgsub*g,vmin=0,vmax=1000)

#apply gaussian filter
filt = NDIMG.filters.gaussian_filter(bgsub*g,3)
s=np.array([1024,1024])
bgsub_rs = IU.DiffReshape(bgsub*g,cxy,s)
filt2s = IU.DiffReshape(filt*cm,cxy,s)
imshow_z(filt2s,vmin=0,vmax=1000)

##########################################################################
#Find basis
##########################################################################
#autocorrelation
autoc = IU.DiffAutoCorr(filt,s,cxy=cxy)
#imshow_z(autoc)

#extract peak frequencies
smallcm = IU.Circularmask(autoc,s/2,3)
pkfft = IU.PeakFilteredFFT(autoc,cxy,sigma=50)
pkfft_abs = np.abs(pkfft)*smallcm
#imshow_z(pkfft_abs)

#get reci lattice vectors
pos = IU.FindPeakPos(pkfft_abs)
pos_zero = pos[:,3:5] - s/2
pos_zero = np.round(pos_zero,2)
bs = VF.find_basis_set(pos_zero.T)
r1,r2 = VF.get_reci_vecs2(bs[:,0],bs[:,1],1024)

fig1 = imshow_z(filt2s,vmin=0,vmax=350)
IU.draw_circles_along_vec(fig1,r1,[512,512],[1024,1024])
IU.draw_circles_along_vec(fig1,r2,[512,512],[1024,1024])

##########################################################################
#Extract peaks
##########################################################################
#sort the peaks by strength and recenter some of them
#thevecs1 = fill_space_with_basis2D(autoc,r1,r2,PY.array([512,512]))
thevecs1 = VF.fill_space_with_basis2D_2(autoc,r1,r2,np.array([512,512]))
#fig1 = imshow_z(filt2s,vmin=0,vmax=250)
#IU.draw_circles_at_vec(fig1,thevecs1)

#slightly shift the centroids and collec the intensities
cents,sums = IU.DiffPeakSums(filt2s,thevecs1)
fig1 = imshow_z(filt2s,vmin=0,vmax=250)
IU.draw_circles_at_vec(fig1,thevecs1,color='w')
IU.draw_circles_at_vec(fig1,cents,size=22,color='r')

#scale vectors to produce data to compare to simulations
camera_length_f = 0.002
nullrad = 0.15
Fvalskspace = np.zeros((np.shape(thevecs1)[0],3))
Fvalskspace[:,0] = (thevecs1[:,0] - 512)*(camera_length_f)
Fvalskspace[:,1] = (thevecs1[:,1] - 512)*(camera_length_f)
rads = np.sqrt(Fvalskspace[:,0]*Fvalskspace[:,0] + Fvalskspace[:,1]*Fvalskspace[:,1])
Fvalskspace[:,2] = sums
Fvalskspace = Fvalskspace[rads>nullrad,:] 

#PLT.close("all")

#############################################################################
#auto find oritentation
#############################################################################

param = np.array([9.8220,14.8030,11.4870,92.7200,80.8700,47.9900]);
cords = np.loadtxt(open("EDO_coors.csv","rb"),delimiter=",");
aform = np.loadtxt(open("EDO_aform.csv","rb"),delimiter=",");
V,Vr = CALG.get_cell_vecs(param)
lambd = 0.0381;
rthick = 0.05;
K0 = 1
B0 = 0
Shapewid = 0.01
Beamwid = 0.10
#uvwg = np.array([0,0,1])
uvwg = np.array([-1.9172,0.7983,3.000])
HKL = VS.getFBall(param,1,cords,aform)


VS.plotfball(HKL)

Fexp = CALG.rotatePattern2D(Fvalskspace,0.45)
VS.plotF(Fexp)

Fsim = VS.getProj(HKL,uvwg,V,Vr,rthick,K0,B0,Shapewid,Beamwid,lambd)
VS.plotProj(HKL,uvwg,V,Vr,rthick,K0,B0,Shapewid,Beamwid,lambd)
VS.plot2F(Fexp,Fsim)

CALG.PearsonCorr2D_fast(Fexp,Fsim)

Fsim_rot = CALG.rotatePattern2D(Fsim,0.758)
VS.plot2F(Fexp,Fsim_rot)
CALG.PearsonCorr2D_fast(Fexp,Fsim_rot)

xx,yy = np.meshgrid(np.arange(-2,2,0.1),np.arange(-2,2,0.1))
zz= np.load('EDOglobalsol1.npy')
VS.plotsurf(xx,yy,zz)

#x0 = np.array([-0.6,0.25,0.70])
#SCO.fmin(VS.Efunglobal,x0,args=(HKL,Fexp),maxfun=10000)