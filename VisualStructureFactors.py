# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 17:17:26 2014

@author: Ray
"""
import CrystalAlg_C as CALG
#from CrystalAlg import *
import numpy as np
from UtilFuns import *
import scipy.optimize as sco
import matplotlib.pyplot as PLT
import matplotlib.cm as cm

from mpl_toolkits.mplot3d import Axes3D

#param = np.array([9.8220,14.8030,11.4870,92.7200,80.8700,47.9900]);
#cords = np.loadtxt(open("EDO_coors.csv","rb"),delimiter=",");
#aform = np.loadtxt(open("EDO_aform.csv","rb"),delimiter=",");
#V,Vr = CALG.get_cell_vecs(param)
#lambd = 0.0381;
#rthick = 0.05;
#K0 = 1
#B0 = 0
#Shapewid = 0.01
#Beamwid = 0.10
#uvw = np.array([-1.9172,0.7983,3.000])
#uvw2 = np.array([0,1,1])
#uvw3 = np.array([0,0,1])
#uvw4 = np.array([1.22,1.5,1.2])

#r1 = np.array([-20.19744833, -41.03271081])*0.002
#r2 = np.array([-27.6386135 ,  63.61605655])*0.002



def getFBall(param,res3D,cords,aform):
    HKL = CALG.simFBall(param,res3D,cords,aform)
    zeroorder_ind1 = np.nonzero((HKL[:,0]==0) & (HKL[:,1]==0) & (HKL[:,2]==0))[0]
    HKL_p = np.delete(HKL,zeroorder_ind1,axis=0)
    return HKL_p

def getProj(HKL,uvw,V,Vr,rthick,K0,B0,Shapewid,Beamwid,lambd):
    HKL2 = CALG.HKL_SLICE(HKL,uvw,V,Vr,rthick)
    kproj = CALG.SPOT_XY3D(HKL2,uvw,V,Vr)    
    fcor = CALG.F_CORR(HKL2[:,3],HKL2[:,4],kproj,K0,B0,Shapewid,Beamwid,lambd)
    return np.column_stack((kproj[:,0:2],fcor))
    
def plotProj(HKL,uvw,V,Vr,rthick,K0,B0,Shapewid,Beamwid,lambd):
    HKL2 = CALG.HKL_SLICE(HKL,uvw,V,Vr,rthick)
    kproj = CALG.SPOT_XY3D(HKL2,uvw,V,Vr)
    fcor = CALG.F_CORR(HKL2[:,3],HKL2[:,4],kproj,K0,B0,Shapewid,Beamwid,lambd)
    
    
    zeroorder_ind1 = np.nonzero((HKL[:,0]==0) & (HKL[:,1]==0) & (HKL[:,2]==0))[0]
    HKL_p = np.delete(HKL,zeroorder_ind1,axis=0)
    
    zeroorder_ind2 = np.nonzero((HKL2[:,0]==0) & (HKL2[:,1]==0) & (HKL2[:,2]==0))[0]
    #HKL2_p = np.delete(HKL2,zeroorder_ind2,axis=0)    
    kproj_p = np.delete(kproj,zeroorder_ind2,axis=0)
    fcor_p = np.delete(fcor,zeroorder_ind2,axis=0)
    fig = PLT.figure()
    PLT.scatter(kproj_p[:,0],kproj_p[:,1],s=fcor_p/2500)
    #plt.gray()
    Fsim = np.empty((np.shape(kproj_p)[0],3))
    Fsim[:,0:2] = kproj_p[:,0:2]
    Fsim[:,2] = fcor_p
    return Fsim

def plotfball(HKL):
    zeroorder_ind1 = np.nonzero((HKL[:,0]==0) & (HKL[:,1]==0) & (HKL[:,2]==0))[0]
    HKL_p = np.delete(HKL,zeroorder_ind1,axis=0)
    fig = PLT.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(HKL_p[:,6],HKL_p[:,7],HKL_p[:,8],s=HKL_p[:,4]/1000)
    ax.set_xlabel('K_x')
    ax.set_ylabel('K_y')
    ax.set_zlabel('K_z')
    ax.set_xlim([-1.0,1.0])
    ax.set_ylim([-1.0,1.0])
    ax.set_zlim([-1.0,1.0])
    PLT.show()
#    n = 100
#    for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
#        xs = randrange(n, 23, 32)
#        ys = randrange(n, 0, 100)
#        zs = randrange(n, zl, zh)
#        ax.scatter(xs, ys, zs, c=c, marker=m)
#

def getDataCsv(filename):
    import itertools
    import csv
    R = np.array([])
    with open(filename, 'rb') as csvfile:
        reader1,reader2  = itertools.tee(csv.reader(csvfile, delimiter=',', lineterminator='\r\n'))
        columns = len(next(reader1))
        R = np.empty((0,columns))
        del reader1
        for row in reader2:
            a = np.array(row).astype(np.float)
            R = np.vstack((R,a))
        del reader2
    return R

def plotF(F,**kwargs):
    color = kwargs.get('color',None)
    if color is None:
        color = 'b'
    PLT.figure()
    PLT.scatter(F[:,0],F[:,1],s=F[:,2]/np.max(F)*200,c=color)
    PLT.show()
    
def plot2F(F1,F2):
    PLT.figure()
    PLT.scatter(F1[:,0],F1[:,1],s=F1[:,2]/np.max(F1)*200,c='b')
    PLT.scatter(F2[:,0],F2[:,1],s=F2[:,2]/np.max(F2)*200,c='r')
    PLT.show()
    
def plot2F_rot(F1,F2,**kwargs):
    theta = kwargs.get('theta',None)
    if theta is None:
        maxcor, theta = calcMaxCor(F1,F2) 
    F2 = CALG.rotatePattern2D(F2,theta)
    PLT.figure()
    PLT.scatter(F1[:,0],F1[:,1],s=F1[:,2]/np.max(F1)*200,c='b')
    PLT.scatter(F2[:,0],F2[:,1],s=F2[:,2]/np.max(F2)*200,c='r')
    PLT.show()

def plotFexpOpt(HKL,F1,x0):
    uvwg = np.array([x0[0],x0[1],1])
    theslice = getProj(HKL,uvwg)
    F2 = rotatePattern2D(theslice,x0[2])
    PLT.figure()
    plt.scatter(F1[:,0],F1[:,1],s=F1[:,2]/np.max(F1)*200,c='b')
    plt.scatter(F2[:,0],F2[:,1],s=F2[:,2]/np.max(F2)*200,c='r')
    plt.show()
    
def plotCorrelations(F1,F2):
    
    rots = arange(0,181,1)
    nrots = np.shape(rots)[0]
    corrs = np.zeros((nrots,))
    for i in range(0,nrots):
        #corrs[i] = PearsonCorr2D_fast(F1,rotatePattern2D(F2,np.pi*rots[i]/180))
        corrs[i] = CALG.PearsonCorr2DPoints_nonvec(F1,CALG.rotatePattern2D(F2,np.pi*rots[i]/180))
    PLT.figure()
    PLT.plot(rots,corrs)
    PLT.show()

def calcMaxCor(F1,F2):
    rots = np.arange(0,181,1)
    nrots = np.shape(rots)[0]
    corrs = np.zeros((nrots,))
    maxval = -inf
    maxtheta = 0
    for i in range(0,nrots):
        corrs[i] = CALG.PearsonCorr2D_fast(F1,CALG.rotatePattern2D(F2,np.pi*rots[i]/180))
        if(corrs[i]>maxval):
            maxval = corrs[i]
            maxtheta = np.pi*rots[i]/180
        #corrs[i] = PearsonCorr2DPoints_nonvec(F1,rotatePattern2D(F2,np.pi*rots[i]/180)
    return maxval,maxtheta

def UVWsurface(HKL,Fexp):
    Hspace = np.arange(-2,2,0.1)
    Kspace = np.arange(-2,2,0.1)
    xx,yy = np.meshgrid(Hspace,Kspace)
    zz = np.zeros((np.shape(Hspace)[0],np.shape(Kspace)[0]))
    tic()
    for i in range(0,np.shape(Hspace)[0]):
        for j in range(0,np.shape(Kspace)[0]):
            uvwg = np.array([Hspace[i],Kspace[j],1])
            zz[i,j],theta = calcMaxCor(Fexp,getProj(HKL,uvwg))
    toc()
    return xx,yy,zz
            
Fexp = getDataCsv('Fexpdata.csv')
Fexp = CALG.rotatePattern2D(Fexp,0.45)

#PearsonCorr(Fexp ,Fexp )
def plotsurf(xx,yy,zz):
    fig = PLT.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx, yy, zz, rstride=1, cstride=1,cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('U')
    ax.set_ylabel('V')
    ax.set_zlabel('Pearson Correlation')    
    PLT.show()

def Efunglobal(x,*args):
    HKL = args[0]
    Fexp = args[1]
    uvwg = np.array([x[0],x[1],1])
    theslice = getProj(HKL,uvwg)
    return -CALG.PearsonCorr2D_fast(Fexp,CALG.rotatePattern2D(theslice,x[2]))
    
def Efunangle(x,*args):
    HKL = args[0]
    Fexp = args[1]
    Fsim = args[2]
    return -CALG.PearsonCorr2D_fast(Fexp,CALG.rotatePattern2D(Fsim,x))
    
#HKL = getFBall()
#Fsim = getProj(HKL,uvw)
#Fsim_rot = rotatePattern2D(Fsim,0.758)
#plot2F(Fexp,Fsim_rot)
#plt.figure()
##Fsim2 = getProj(HKL,array([-0.6,0.2,1]))
#sco.fmin(Efunangle,0.7,args=(HKL,Fexp,Fsim))
#x0 = np.array([-0.6,0.25,0.70])
#sco.fmin(Efunglobal,x0,args=(HKL,Fexp),maxfun=10000)

#l1 = np.array([-2,-2,-pi/2])
#u1 = np.array([2,2,pi/2])
#sco.anneal(theerrorfunction,x0,np.array([0,0,0.7]),maxeval=100000,maxaccept=-0.8,lower=l1,upper=u1)
#sco.fmin_bfgs(theerrorfunction,x0,gtol=1e-06)
#sco.fmin_powell(theerrorfunction,x0)
#def teste2(F1,F2):
#    #for i in range(0,100):
#    rang = np.arange(-1,1,0.05)
#    binshape = np.shape(rang)[0] +1 
#    x1 = np.digitize(F1[:,0],rang)
#    y1 = np.digitize(F1[:,1],rang)
#    x2 = np.digitize(F2[:,0],rang)
#    y2 = np.digitize(F2[:,1],rang)
#    bins1 = np.zeros((binshape,binshape))
#    bins2 = np.zeros((binshape,binshape))
#    bins1[x1,y1] = F1[:,2]
#    bins2[x2,y2] = F2[:,2]
#    lininds1 = (x1)*binshape+(y1)
#    lininds2 = (x2)*binshape+(y2)        
#    h1 = histogram(lininds1,arange(0,(binshape)*(binshape)+1))[0]
#    h2 = histogram(lininds2,arange(0,(binshape)*(binshape)+1))[0]
#    hp1 = h1.reshape((binshape,binshape))
#    hp2 = h2.reshape((binshape,binshape))
#    bins1 = bins1*hp1
#    bins2 = bins2*hp2
#    return PearsonCorr(bins1,bins2)

    

#HKL = getFBall()
#Fsim = getProj(HKL,uvw)
#Fsim = plotProj(HKL,uvw)
#
#Fsim_rot = rotatePattern2D(Fsim,0.24)
#PearsonCorr2DPoints_nonvec(Fexp,Fsim_rot)
    

#import csv
#myfile = open('test1.csv', 'wb')
#wr = csv.writer(myfile, delimiter=',',lineterminator='\r\n')
#for i in range(0,shape(HKL2)[0]):
#    wr.writerow(HKL[i,:])
#myfile.close()

#def teste():
#    tic()
#    for i in range(0,100):
#        Fsim = getProj(HKL,uvw)
#        PearsonCorr2DPoints_nonvec(Fexp,Fsim)
#    toc()