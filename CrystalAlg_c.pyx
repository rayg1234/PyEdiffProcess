# -*- coding: utf-8 -*-
# cython: profile=True
# filename: CrystalAlg_c.pyx
"""
Created on Thu Mar 27 17:22:04 2014

Crystal Algebric functions:
This set of functions deals with simulating reciprocal lattice points given the 
crystal parameters and the atomic form factors for electrons. The simulations
uses the kinematic approach to electron diffraction. 

functions get_cell_vecs,simFball,calc_SFs,HKL_SLICE,SPOT_XY3D are used for the
kinematic simulations

The rest of the functions are different version of pearson correlation coefficient 
calculations for comparing experimental structure factors to simulated ones.
The fastest version by far is PearsonCorr2D_fast but uses a vectorized digitization
algorithm to approximate the solutions and maybe prone to errors if the mesh size
is set too small.


@author: Ray Gao, 2014
"""

import numpy as np
import math
cimport numpy as np
import numpy.linalg as lp
import VectorFuns as VF
#from UtilFuns import *
DTYPE = np.double
DTYPE2 = np.int
ctypedef np.double_t DTYPE_t
ctypedef np.int_t DTYPE2_t


"""
get_cell_vecs computes the real lattice and reciprocal lattice vectors in cartesian coords where the
vector a is parallel to x direction
[Inputs]
params: an 1x6 array containing the cell parameters [a,b,c,alpha,beta,gamma] angles are in degrees

[Returns]
V: real lattice vectors [a,b,c], a is parallel to x direction
Vr: reciprocal lattice vectors [a*,b*,c*]

"""
def get_cell_vecs(params):
    
    a = params[0];
    b = params[1];
    c = params[2];
    alpha = params[3]*np.pi/180;
    beta = params[4]*np.pi/180;
    gamma = params[5]*np.pi/180;

    #lattice vectors
    aV = np.array([a,0,0]);
    bV = np.array([b*np.cos(gamma),b*np.sin(gamma),0]);
    cV = np.array([c*np.cos(beta), c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma), c*np.sqrt(1-np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2+2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))/np.sin(gamma)]);
    
    Omega = np.abs(np.dot(aV,np.cross(bV,cV)));
    
    #recip lattice vectors
    aVr = np.cross(bV,cV)/Omega;
    bVr = np.cross(cV,aV)/Omega;
    cVr = np.cross(aV,bV)/Omega;
    
    V = np.array([aV,bV,cV]).T;
    Vr = np.array([aVr,bVr,cVr]).T;
    return V,Vr;
    
"""
simFBall uses the crystal lattice parameters, atomic coordinates, and electron atomic form factors to
generate a 3D ball of structure factors within the given resolution limit. Note this function is not 
vectorized currently.

[Inputs]
params: an 1x6 array containing the cell parameters [a,b,c,alpha,beta,gamma] angles are in degrees
resol3D: the desired resolution limit
coors: an nx6 array of atomic coordinates with 6 cols [atom #,x,y,z,uiso,occupancy]
aform: an nx11 array of atomic form factor coefficients using according Peng's formula for
    approximating form factors (L.M. Peng et. al Acta Crys. A, 1996). [atom #,a1...a5,b1...b5]

[Returns]
HKL: HKL vector, m x 9 array where m is the number of structure factors allowed in the res. limit,
the columns are [H,K,L,d*,SF_amplitude,SF_phase,kx,ky,kz]
H,K,L are the miller indices
d* is the recipricol spacing
SF_amplitude and SF_phase are the structure factor amps and factors respectively
kx,ky,kz are the coords in k-space

"""
def simFBall(params,int resol3D,np.ndarray[DTYPE_t, ndim=2] coors,np.ndarray[DTYPE_t, ndim=2] aform):
    
    V_t,Vr_t = get_cell_vecs(params)
    cdef np.ndarray[DTYPE_t, ndim=2] V = V_t
    cdef np.ndarray[DTYPE_t, ndim=2] Vr = Vr_t
    
    #generate HKL sphere
    #tic();
    cdef int HMAX = 2.0*np.ceil(resol3D/lp.norm(Vr[:,0]))
    cdef int KMAX = 2.0*np.ceil(resol3D/lp.norm(Vr[:,1]))
    cdef int LMAX = 2.0*np.ceil(resol3D/lp.norm(Vr[:,2]))
    cdef np.ndarray[DTYPE_t, ndim=2] HKL = np.zeros([(2*HMAX+1)*(2*KMAX+1)*(2*LMAX+1),9],dtype=DTYPE)
    cdef int i = 0
    cdef int h,k,l
    for h in range(np.int(-HMAX),np.int(HMAX)+1):
        for k in range(np.int(-KMAX),np.int(KMAX)+1):
            for l in range(np.int(-LMAX),np.int(LMAX)+1):
                #dk = np.linalg.norm(h * Vr[:,0] + k * Vr[:,1] + l * Vr[:,2]);
                #if(dk <= resol3D):
                HKL[i,0] = h;
                HKL[i,1] = k;
                HKL[i,2] = l;
                #HKL[i,3] = dk;
                i = i+1;
    cdef np.ndarray[DTYPE_t, ndim=2] y = np.dot(Vr,HKL[:,0:3].T).T
    cdef np.ndarray[DTYPE_t, ndim=1] thenorms = np.sqrt(y[:,0]**2 + y[:,1]**2 + y[:,2]**2)
    HKL = HKL[thenorms<=resol3D,:]
    HKL[:,3] = thenorms[thenorms<=resol3D]

        
    
    NUMREFS = np.shape(HKL)[0]
    
    cdef DTYPE_t sf = 0
    for i in xrange(0,NUMREFS):
        sf = calc_SFs(HKL[i,0],HKL[i,1],HKL[i,2],HKL[i,3],coors,aform)
        HKL[i,4] = abs(sf)*1000
        HKL[i,5] = np.sign(sf)
    #HKL = HKL[0:i,:];
    dk = np.dot(Vr_t,HKL[:,0:3].T)
    HKL[:,6:9] = dk.T
    
    #toc();
    return HKL;
        
"""
calc_SFs computes the structure factors given the miller index, the atomic coords, and the 
atomic form factors. It uses Peng's (L.M. Peng et. al Acta Crys. A, 1996) method of sum of Gaussians
to interpolate the form factors.

[Inputs]
h,k,l: miller indices as integers
coords: an nx6 array of atomic coordinates with 6 cols [atom #,x,y,z,uiso,occupancy]
aform: aform: an nx11 array of atomic form factor coefficients using according Peng's formula for
    approximating form factors (L.M. Peng et. al Acta Crys. A, 1996). [atom #,a1...a5,b1...b5]

[Returns]
F: The structure factor (a real number). The phase is either 0 for positive, and 180 deg for negative)
"""    
def calc_SFs(int h,int k,int l,double s,np.ndarray[DTYPE_t, ndim=2] coords,np.ndarray[DTYPE_t, ndim=2] aform):
    cdef int natom = np.shape(coords)[0]
    cdef DTYPE_t F = 0  #structure factors
    cdef DTYPE_t f = 0  #atomic form factors
    cdef int i,j = 0
    cdef DTYPE_t az = 0
    cdef DTYPE_t hr = 0
    for i in range(0,natom):
        hr = h*coords[i,1] + k*coords[i,2] + l*coords[i,3]
        az = coords[i,0]
        f = 0
        for j in range(0,5):
            f = f + aform[az-1,j+1]*math.exp(-aform[az-1,j+1+5]*(s*s))
        F = F + coords[i,5]*f*2*math.cos(2*math.pi*hr)
    return F

"""
HKL_SLICE (Vectorized) computes a slice of the structure factor sphere. The slice has thickness 2*rThick, 
and is perpendicular to direction [uvw]

[Inputs]
HKL: miller indces vector (mx9 array returned from simFBall)
uvw: vector describing the direction perpendicular to the slice
V: lattice vectors given by get_cell_vecs
Vr: reciprocol lattice vectors given by get_cell_vecs
rThick: thickness of the slice in angstroms (ie: 0.05)

[Returns]
HKL: an HKL vector with same column dimensions as that returned by simFBall but with rows reduced to only
those intercepting the requested slice.
"""
def HKL_SLICE(HKL,uvw,V,Vr,DTYPE_t rThick):
    uvwN = np.dot(V,uvw)
    uvwN = uvwN/np.sqrt(uvwN[0]**2 + uvwN[1]**2 + uvwN[2]**2)
    dk = np.dot(Vr,HKL[:,0:3].T)
    t = np.abs(np.dot(dk.T,uvwN))
    
    return HKL[t<rThick,:]

"""
#SPOT_XY3D (Vectorized) project the onto plane normal to UVW vector
HKL: miller indces vector (mx9 array returned from simFBall)
uvw: vector describing the direction perpendicular to the slice
V: lattice vectors given by get_cell_vecs
Vr: reciprocol lattice vectors given by get_cell_vecs

[Returns]
kproj: an nx3 vector contains the projected coordinates on the plane normal to the UVW vector, [kx_proj,ky_proj,kz_proj]
"""
def SPOT_XY3D(HKL,uvw,V,Vr):
    uvwN = np.dot(V,uvw)
    uvwN = uvwN/np.sqrt(uvwN[0]**2 + uvwN[1]**2 + uvwN[2]**2)
    k = np.zeros((3,3))
    #define new cart coordiantes
    k[:,2] = -uvwN    
    k[:,1] = np.cross(k[:,2],Vr[:,0])
    k[:,1] = k[:,1]/np.sqrt(k[0,1]**2 + k[1,1]**2 + k[2,1]**2)
    k[:,0] = np.cross(k[:,1],k[:,2])
    dk = np.dot(Vr,HKL[:,0:3].T)
    
    kproj = np.zeros((np.shape(HKL)[0],3))
    kproj[:,0] = np.dot(dk.T,k[:,0])
    kproj[:,1] = np.dot(dk.T,k[:,1])
    kproj[:,2] = np.dot(dk.T,k[:,2])
    #k[:,1] = np.cross(uwvN)
    return kproj


"""
F_CORR (Vectorized) applies a serious of corrections to the structure factors based on realistic experimental parameters. 
The corrections includes a temperature factor, shape factor and beam divergence.
See (Gao et. al Nature 496 2013 Supplementary Info.) for more details.

[Inputs]
d3d: the recirpocal lattice space in 3D (same as d* in the HKL vector from simFBall)
F: the structure factor amplitudes
kproj: the return from SPOT_XY3D
k0, B0, Shapewid, Beamwid: correction factors from the experiment
lambd: the electron beam wavelength

[Returns]
Fcorr: an array containing the corrected structure factor amplitudes
"""
def F_CORR(d3d,F,kproj,K0,B0,Shapewid,Beamwid,lambd):
    #length of kvec in 2D
    d2d = np.sqrt(kproj[:,0]**2 + kproj[:,1]**2)
    dsewald = kproj[:,2] - (1/lambd - np.cos(np.arcsin((kproj[:,0]**2+kproj[:,1]**2)*lambd))/lambd)
    
    TempFac = K0*np.exp(-B0*d3d**2)
    Sigma = np.sqrt((d2d*Beamwid)**2 + Shapewid**2)
    ShapeFac = (1.0/Sigma)*np.exp(-(kproj[:,2]**2)/(2*Sigma**2))
    
    Fcorr = np.sqrt(ShapeFac)*TempFac*F
    return Fcorr

"""
find_Candidates attempts to find a series of guess orientations to begin orientation matching between the
experimental and the simulated pattern. The user supplies 2 basis vectors that span the spacce of the measured
2D diffraction patterns. The guess candidates are obtained by applying the laue equation h*u + k*v + l*w = 0 for the 0th order zone.
It returns the set of UVW within some the specified tolerance that satisfies the laue equation.

[Inputs]
HKL: miller indces vector (mx9 array returned from simFBall)
r1,r2: two basis vectors that span the space of the 2D experimental diffraction pattern
optional args
tol: the distance tolerance (default: 0.02 inverse angstrom)
atol: the angular tolerance (default: 0.2 rad)

if the user is not confident about the camera length then the distance tolerance should be increased.

[Returns]
uvwguess: an nx3 array containing a set of UVW guesses to being orientation matching
"""
def find_Candidates(HKL,r1,r2,**kwargs):
    tol = kwargs.get('tol',None) #distance tolerance
    atol = kwargs.get('atol',None) #angle tolerance
    if tol is None:
        tol = 0.02
    if atol is None:
        atol = 0.2
    d1 = math.sqrt(r1[0]*r1[0] + r1[1]*r1[1])
    d2 = math.sqrt(r2[0]*r2[0] + r2[1]*r2[1])
    theta = VF.angle_btw_vecs(r1,r2)
    #first find indices close to d1
    inds1 = np.nonzero( (HKL[:,3]<d1+tol) & (HKL[:,3]>d1-tol))[0]
    inds2 = np.nonzero( (HKL[:,3]<d2+tol) & (HKL[:,3]>d2-tol))[0]
    s1 = np.shape(inds1)[0]
    s2 = np.shape(inds2)[0]
    angles = np.zeros((s1*s2,3))
    for i in range(0,s1):
        for j in range(0,s2):
            cind = i*s2+j
            angles[cind,0] = inds1[i]
            angles[cind,1] = inds2[j]
            angles[cind,2] = VF.angle_btw_vecs(HKL[inds1[i],6:9],HKL[inds2[j],6:9])
    inds3 = np.nonzero( (angles[:,2]< theta + atol) & (angles[:,2]> theta - atol) )[0]
    cands = angles[inds3]
    uvwguess = np.zeros((np.shape(inds3)[0],3))
    for i in range(0,np.shape(inds3)[0]):
        h1 = HKL[cands[i,0],0]
        k1 = HKL[cands[i,0],1]
        l1 = HKL[cands[i,0],2]
        h2 = HKL[cands[i,1],0]
        k2 = HKL[cands[i,1],1]
        l2 = HKL[cands[i,1],2]
        uvwguess[i,0] = k1*l2 - l1*k2
        uvwguess[i,1] = -(h1*l2 - l1*h2)
        uvwguess[i,2] = h1*k2 - k1*h2
        
    #compress candidates
    #to be implemented ...        
    return uvwguess,cands

"""
PearsonCorr2D_fast (Vectorized) is a completely vectorized implementation to compute the 2D Pearson Correlation Coefficient
between the simulated and experimental diffraction patterns. The implementation uses the numpy digitization function
to bin the inputs points into an m x m 2D (m = binwid) image and then applies the pearson correlation function
to the 2D image itself.

[Inputs]
F1,F2: these are the sets of diffraction patterns to compare. They are given as a nx3 array where
the first two columns represent the 2D coordinates in k-space and the last is the structure factor amp 
[k_x,k_y,SF]

optional arg:
binwid: the widths of the bins in inverse angstroms. If the binwid is set too large then many spots
will be grouped together and the function will be inaccurate. If the binwid is set too small then
the computation will be slower. The user should visually inspect the bins first to make sure the binsize
is appropriate.

[Returns] 
a single number representing the pearson correlation between the diffraction patterns.
"""
def PearsonCorr2D_fast(F1,F2,**kwargs):
    binwid = kwargs.get('binwid',None)
    if binwid is None:
        binwid =  0.05
    rang = np.arange(-1,1,binwid)
    binshape = np.shape(rang)[0] +1 
    x1 = np.digitize(F1[:,0],rang)
    y1 = np.digitize(F1[:,1],rang)
    x2 = np.digitize(F2[:,0],rang)
    y2 = np.digitize(F2[:,1],rang)
    bins1 = np.zeros((binshape,binshape))
    bins2 = np.zeros((binshape,binshape))
    bins1[x1,y1] = F1[:,2]
    bins2[x2,y2] = F2[:,2]
    lininds1 = (x1)*binshape+(y1)
    lininds2 = (x2)*binshape+(y2)        
    h1 = np.histogram(lininds1,np.arange(0,(binshape)*(binshape)+1))[0]
    h2 = np.histogram(lininds2,np.arange(0,(binshape)*(binshape)+1))[0]
    hp1 = h1.reshape((binshape,binshape))
    hp2 = h2.reshape((binshape,binshape))
    bins1 = bins1*hp1
    bins2 = bins2*hp2
    return PearsonCorr(bins1,bins2)

"""
PearsonCorr2DPoints_nonvec2 is slower way of computing the pearson correlation coefficient between two
diffraction patterns. This function just tries to pair spots within some distance of each other. This
function is nearly identical to PearsonCorr2DPoints_nonvec except it doesn't count the points with 0
intensity. PearsonCorr2D_fast should always be used instead unless for debugging purposes.

[Inputs]
F1,F2: these are the sets of diffraction patterns to compare. They are given as a nx3 array where
the first two columns represent the 2D coordinates in k-space and the last is the structure factor amp 
[k_x,k_y,SF]

[Returns] 
a single number representing the pearson correlation between the diffraction patterns.
"""
def PearsonCorr2DPoints_nonvec2(np.ndarray[DTYPE_t, ndim=2] F1,np.ndarray[DTYPE_t, ndim=2] F2):
    cdef DTYPE_t thresh = 0.03**2    
    cdef np.ndarray[DTYPE_t, ndim=1] F2prime = np.zeros((np.shape(F1)[0]),dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] F1prime = np.zeros((np.shape(F1)[0]),dtype=DTYPE)
    cdef int nzcount = 0
    for i in range(0,np.shape(F1)[0]):
        ds = (F1[i,0] - F2[:,0])*(F1[i,0] - F2[:,0]) + (F1[i,1] - F2[:,1])*(F1[i,1] - F2[:,1])
        s = np.sum(F2[ds<thresh,2])
        if(s!=0):
            F2prime[nzcount] = s
            F1prime[nzcount] = F1[i,2]
            nzcount = nzcount+1
    return PearsonCorr(F1prime[0:nzcount+1],F2prime[0:nzcount+1])

"""
PearsonCorr2DPoints_nonvec is slower way of computing the pearson correlation coefficient between two
diffraction patterns. This function just tries to pair spots within some distance of each other. PearsonCorr2D_fast
should always be used instead unless for debugging purposes.

[Inputs]
F1,F2: these are the sets of diffraction patterns to compare. They are given as a nx3 array where
the first two columns represent the 2D coordinates in k-space and the last is the structure factor amp 
[k_x,k_y,SF]

[Returns] 
a single number representing the pearson correlation between the diffraction patterns.
"""
def PearsonCorr2DPoints_nonvec(np.ndarray[DTYPE_t, ndim=2] F1,np.ndarray[DTYPE_t, ndim=2] F2):
    cdef DTYPE_t thresh = 0.03**2    
    cdef np.ndarray[DTYPE_t, ndim=1] F2prime = np.zeros((np.shape(F1)[0]),dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] ds = np.zeros((np.shape(F1)[0]),dtype=DTYPE)
    for i in range(0,np.shape(F1)[0]):
        ds = (F1[i,0] - F2[:,0])*(F1[i,0] - F2[:,0]) + (F1[i,1] - F2[:,1])*(F1[i,1] - F2[:,1])
        F2prime[i] = np.sum(F2[ds<thresh,2])
    return PearsonCorr(F1[:,2],F2prime)

"""
rotatePattern2D (Vectorized) rotates a diffraction pattern by some angle theta

[Inputs]
F: the diffraction pattern to rotate. A nx3 array where
the first two columns represent the 2D coordinates in k-space and the last is the structure factor amp 
[k_x,k_y,SF]
theta: the angle to rotate in radians

[Returns] 
the rotated diffraction pattern
"""
def rotatePattern2D(F,theta):
    Rot = [[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]]
    temp = np.dot(Rot,F[:,0:2].T).T
    return np.column_stack((temp,F[:,2]))

"""
PearsonCorr (Vectorized) computes the pearson correlation coefficient of two set of points. The number of points
in the sets must be equal. 

[Inputs]
vals1, val2s: arrays containing equal number of values

[Returns] 
The pearson correlation coefficient.
+1 means perfect correlation. 0 means no correlation. -1 means perfect anti-correlation
"""
def PearsonCorr(vals1,vals2):
    m1 = np.mean(vals1)
    m2 = np.mean(vals2)
    
    c1 = (vals1 - m1)/m1
    c2 = (vals2 - m2)/m2
    gamma = np.sum(c1*c2)/(np.sqrt(np.sum(c1*c1))*np.sqrt(np.sum(c2*c2)))

    if(gamma==np.nan):
        gamma = 0    
    return gamma

"""
Depricated function for computing form factors
"""
def calc_aform(DTYPE_t s,DTYPE_t az,np.ndarray[DTYPE_t, ndim=2] aform):
    cdef DTYPE_t f = 0
    cdef int i = 0
    for i in range(0,5):
        f = f + aform[az-1,i+1]*math.exp(-aform[az-1,i+1+5]*(s*s))
        
    #print f
    return f
    
