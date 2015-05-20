#!/usr/bin/env python

#*******************************
# A quick output of quicksim simulation for DESI
# For each 500 input spectra, this wrapper Outputs Four files for each arm B,R,Z,
#	1. frame file (x3)
#	2. Flux Calibration Vector file (x3)
#	3. Sky Model File (x3)
#	4. cframe file (x3)
#
#Example:
#
#********************************

import argparse
import os
import os.path
import numpy as np
import scipy.sparse as sp
import astropy.io.fits as pyfits
import sys
import desimodel.simulate as sim
import desispec
import desisim.io
from desispec.resolution import Resolution
from desispec.io import write_flux_calibration, write_fiberflat
from desispec.interpolation import resample_flux


#- Parse arguments
parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#      usage = "%prog [arguments]"
#      )


parser.add_argument("--input",type=str, help="input spectra")
parser.add_argument("--fiberfile",type=str, help='fiber map file')
parser.add_argument("--exptime", type=int, help="exposure time")
parser.add_argument("--nspectra",type=int,default=500,help='no. of spectra to be simulated, starting from first')
parser.add_argument("--nstart", type=int, default=0,help='starting spectra # for simulation')
parser.add_argument("--airmass",type=float, help="airmass")
args = parser.parse_args()

#must import desimodel
if 'DESIMODEL' not in os.environ:
    raise RuntimeError('The environment variable DESIMODEL must be set.')
DESIMODEL_DIR=os.environ['DESIMODEL'] 


# Look for Directory tree/ environment set up
# Directory Tree is $DESI_SPECTRO_REDUX/$PRODNAME/exposures/NIGHT/EXPID/*.fits
# Perhaps can be synced with desispec findfile?

#But read fibermap file and extract the headers needed for Directory tree

#read fibermapfile to get objecttype,NIGHT and EXPID....
if args.fiberfile:
 
    print "Reading fibermap file %s"%(args.fiberfile)
    tbdata,hdr=desispec.io.fibermap.read_fibermap(args.fiberfile)
    objtype=tbdata['OBJTYPE'].copy()
    #need to replace STD object types with STAR since quicksim expects star instead of std
    stdindx=np.where(objtype=='STD') # match STD with STAR
    objtype[stdindx]='STAR'
    NIGHT=hdr['NIGHT']
    EXPID=hdr['EXPID']


else:
    print "Need Fibermap file"


#----------DESI_SPECTRO_REDUX--------
DESI_SPECTRO_REDUX_DIR="./quickGen"

if 'DESI_SPECTRO_REDUX' not in os.environ:

    print 'DESI_SPECTRO_REDUX environment is not set.'
    
else:
    DESI_SPECTRO_REDUX_DIR=os.environ['DESI_SPECTRO_REDUX']

if os.path.exists(DESI_SPECTRO_REDUX_DIR):

    if not os.path.isdir(DESI_SPECTRO_REDUX_DIR):
        raise RuntimeError("Path %s Not a directory"%DESI_SPECTRO_REDUX_DIR)
else:
    try:
        os.makedirs(DESI_SPECTRO_REDUX_DIR)
    except:
        raise
        
#---------PRODNAME-----------------

PRODNAME_DIR='prodname'
if 'PRODNAME' not in os.environ:
    print 'PRODNAME environment is not set.'
else:
    PRODNAME_DIR=os.environ['PRODNAME']
prod_Dir=os.path.join(DESI_SPECTRO_REDUX_DIR,PRODNAME_DIR)

if os.path.exists(prod_Dir):

    if not os.path.isdir(prod_Dir):
        raise RuntimeError("Path %s Not a directory"%prod_Dir)
else:
    try:
        os.makedirs(prod_Dir)
    except:
        raise

#----------exposures-------------

#exposure_Dir=os.path.join(prod_Dir,'exposures')
#if os.path.exists(exposure_Dir):
#    if not os.path.isdir(exposure_Dir):
#        raise RuntimeError("Path %s Not a directory"%exposure_DIR)
#else:
#    try:
#        os.makedirs(exposure_Dir)
#    except:
#        raise

#----------NIGHT--------------

#NIGHT_DIR=os.path.join(exposure_Dir,NIGHT)
#if os.path.exists(NIGHT_DIR):
#    if not os.path.isdir(NIGHT_DIR):
#        raise RuntimeError("Path %s Not a directory"%NIGHT_DIR)
#else:
#    try:
#        os.makedirs(NIGHT_DIR)
#    except:
#        raise

#---------EXPID-----------

#EXPID_DIR=os.path.join(NIGHT_DIR,"%08d"%EXPID)
#if os.path.exists(EXPID_DIR):
#    if not os.path.isdir(EXPID_DIR):
#        raise RuntimeError("Path %s Not a directory"%EXPID_DIR)
#else:
#    try:
#        os.makedirs(EXPID_DIR)
#    except:
#        raise
#
#if args.input is None:
#    print('sys.stderr,"ERROR -i/--input filename required"')

#- TODO: make this an option
spectrograph=0

#read the input file (simspec file)
print 'Now Reading the input file',args.input
simspec = desisim.io.read_simspec(args.input)
wavelengths = simspec.wave['brz']
spectra = simspec.flux

# hdulist=pyfits.open(args.input)
# data=hdulist[0].data
# hdr=hdulist[0].header
# wavelengths=hdr["CRVAL1"]+hdr["CDELT1"]*np.arange(len(data[1,:]))
# spectra=data/1.0e-17# flux in units of 1.0e-17 ergs/cm^2/s/A

#print "File Shape:", data.shape
print "wavelength range:", wavelengths[0], "to", wavelengths[-1]
# nspec=data.shape[0]
# nwave=data.shape[1]
nspec = simspec.nspec
nwave = len(simspec.wave['brz'])

# Here we will run for single CCD(500 x 3 in total). Fewer spectra can be run using 'nspectra' and 'nstart' options

print " simulating spectra",args.nstart, "to", args.nspectra+args.nstart-1

print "************************************************"


# Run Simulation for first Spectrum
print "Initializing QuickSim"
specObj0=sim.SpectralFluxDensity(wavelengths,spectra[args.nstart,:])
qsim=sim.Quick(basePath=DESIMODEL_DIR)

# print "Simulating Spectrum",args.nstart," of spectrograph",spectrograph, "object type:", objtype[args.nstart]

#- simulate a fake object 0 just to get wavelength grids etc setup
results=qsim.simulate(sourceType='star',sourceSpectrum=specObj0,airmass=args.airmass,expTime=args.exptime)
observedWavelengths=results.wave
origin_wavelength=qsim.wavelengthGrid
waveLimits={'b':(3569,5949),'r':(5625,7741),'z':(7435,9834)}

#- Check if input simspec is for a continuum flat lamp instead of science
if simspec.flavor == 'flat':
    print "Simulating flat lamp exposure"
    for channel in ('b', 'r', 'z'):
        camera = channel+str(spectrograph)
        outfile = desispec.io.findfile('fiberflat', NIGHT, EXPID, camera)
        ii = (waveLimits[channel][0] <= observedWavelengths) & (observedWavelengths <= waveLimits[channel][1])
        wave = observedWavelengths[ii]
        dw = np.gradient(simspec.wave[channel])
        meanspec = resample_flux(wave, simspec.wave[channel], np.average(simspec.phot[channel]/dw, axis=0))
        fiberflat = np.random.normal(loc=0.0, scale=1.0/np.sqrt(meanspec), size=(nspec, len(wave)))
        ivar = np.tile(1.0/meanspec, nspec).reshape(nspec, len(meanspec))
        mask = np.zeros((simspec.nspec, len(wave)))
        
        write_fiberflat(outfile, fiberflat, ivar, mask, meanspec, wave, header=None)
        print "Wrote "+outfile
    
    sys.exit(0)    

# Define wavelimits for B,R,Z camera(by hand this time), later simulated data is cropped within these limits

# Get the camera ranges from quicksim outputs 
brange,=np.where((observedWavelengths>=waveLimits['b'][0])&(observedWavelengths<=waveLimits['b'][1]))
rrange,=np.where((observedWavelengths>=waveLimits['r'][0])&(observedWavelengths<=waveLimits['r'][1]))
zrange,=np.where((observedWavelengths>=waveLimits['z'][0])&(observedWavelengths<=waveLimits['z'][1]))

# save the observation wavelengths for each camera
bwaves=observedWavelengths[brange]
rwaves=observedWavelengths[rrange]
zwaves=observedWavelengths[zrange]
bmaxbin,=brange.shape
zmaxbin,=zrange.shape
rmaxbin,=rrange.shape
maxbin=max(bmaxbin,zmaxbin,rmaxbin)


# Now break the simulated outputs in three different ranges.  

# Object Photons

nobj=np.zeros((maxbin,results.nobj.shape[1],500))
nobj[:bmaxbin,0,args.nstart]=results.nobj[brange,0]
nobj[:rmaxbin,1,args.nstart]=results.nobj[rrange,1]
nobj[:zmaxbin,2,args.nstart]=results.nobj[zrange,2]

# Sky Photons

nsky=np.zeros((maxbin,3,500))
nsky[:bmaxbin,0,args.nstart]=results.nsky[brange,0]
nsky[:rmaxbin,1,args.nstart]=results.nsky[rrange,1]
nsky[:zmaxbin,2,args.nstart]=results.nsky[zrange,2]

# Inverse Variance (Object+Sky Photons)

nivar=np.zeros((maxbin,3,500))
nivar[:bmaxbin,0,args.nstart]=1/((results.nobj[brange,0])+(results.nsky[brange,0])+(results.rdnoise[brange,0])**2.0+(results.dknoise[brange,0])**2.0)
nivar[:rmaxbin,1,args.nstart]=1/((results.nobj[rrange,1])+(results.nsky[rrange,1])+(results.rdnoise[rrange,1])**2.0+(results.dknoise[rrange,1])**2.0)
nivar[:zmaxbin,2,args.nstart]=1/((results.nobj[zrange,2])+(results.nsky[zrange,2])+(results.rdnoise[zrange,2])**2.0+(results.dknoise[zrange,2])**2.0)

# Inverse Variance in Sky counts

sky_ivar=np.zeros((maxbin,3,500)) # Multiply by 25, decreasing the error by a factor of 5.
sky_ivar[:bmaxbin,0,args.nstart]=25*1/((results.nsky[brange,0])+(results.rdnoise[brange,0])**2.0+(results.dknoise[brange,0])**2.0)
sky_ivar[:rmaxbin,1,args.nstart]=25*1/((results.nsky[rrange,1])+(results.rdnoise[rrange,1])**2.0+(results.dknoise[rrange,1])**2.0)
sky_ivar[:zmaxbin,2,args.nstart]=25*1/((results.nsky[zrange,2])+(results.rdnoise[zrange,2])**2.0+(results.dknoise[zrange,2])**2.0)

# Calibrated Object Flux ( in units of 10^-17 ergs/s/cm2/A)

cframe_observedflux=np.zeros((maxbin,3,500))
cframe_observedflux[:bmaxbin,0,args.nstart]=results.obsflux[brange]
cframe_observedflux[:rmaxbin,1,args.nstart]=results.obsflux[rrange]
cframe_observedflux[:zmaxbin,2,args.nstart]=results.obsflux[zrange]

# Inverse Variance in Object Flux (in units of [10^-17 ergs/s/cm2/A]^-2)

cframe_ivar=np.zeros((maxbin,3,500))
cframe_ivar[:bmaxbin,0,args.nstart]=results.ivar[brange]
cframe_ivar[:rmaxbin,1,args.nstart]=results.ivar[rrange]
cframe_ivar[:zmaxbin,2,args.nstart]=results.ivar[zrange]
	
# Random Gaussian Noise to nobj+nsky

rand_noise=np.zeros((maxbin,3,500))
np.random.seed(0)
rand_noise[:bmaxbin,0,args.nstart]=np.random.normal(np.zeros(bmaxbin),np.ones(bmaxbin)/np.sqrt(nivar[:bmaxbin,0,args.nstart]),bmaxbin)
rand_noise[:rmaxbin,1,args.nstart]=np.random.normal(np.zeros(rmaxbin),np.ones(rmaxbin)/np.sqrt(nivar[:rmaxbin,1,args.nstart]),rmaxbin)
rand_noise[:zmaxbin,2,args.nstart]=np.random.normal(np.zeros(zmaxbin),np.ones(zmaxbin)/np.sqrt(nivar[:zmaxbin,2,args.nstart]),zmaxbin)

# Random Gaussian Noise to SKY only

sky_rand_noise=np.zeros((maxbin,3,500))
sky_rand_noise[:bmaxbin,0,args.nstart]=np.random.normal(np.zeros(bmaxbin),np.ones(bmaxbin)/np.sqrt(sky_ivar[:bmaxbin,0,args.nstart]),bmaxbin)
sky_rand_noise[:rmaxbin,1,args.nstart]=np.random.normal(np.zeros(rmaxbin),np.ones(rmaxbin)/np.sqrt(sky_ivar[:rmaxbin,1,args.nstart]),rmaxbin)
sky_rand_noise[:zmaxbin,2,args.nstart]=np.random.normal(np.zeros(zmaxbin),np.ones(zmaxbin)/np.sqrt(sky_ivar[:zmaxbin,2,args.nstart]),zmaxbin)

#construct resolution matrix from sigma_vs_wavelength. First resample to respective sigmas vs wavelengths

sigma_b_vs_wave=sim.WavelengthFunction(origin_wavelength,qsim.cameras[0].sigma_wave).getResampledValues(bwaves)
sigma_r_vs_wave=sim.WavelengthFunction(origin_wavelength,qsim.cameras[1].sigma_wave).getResampledValues(rwaves)
sigma_z_vs_wave=sim.WavelengthFunction(origin_wavelength,qsim.cameras[2].sigma_wave).getResampledValues(zwaves)

#This definition follows form qsim

def resolution_matrix(sigma_vs_wavelength,wavelengthGrid,throughput):
        nhalf = 50
        nbins = wavelengthGrid.size
        sparseData = np.empty((2*nhalf*nbins,))
        sparseIndices = np.empty((2*nhalf*nbins,),dtype=np.int32)
        sparseIndPtr = np.empty((nbins+1,),dtype=np.int32)
        nextIndex = 0
        for bin in range(nbins):
            sparseIndPtr[bin] = nextIndex
            if bin >= nhalf and bin < nbins-nhalf and throughput[bin] > 0:
                lam = wavelengthGrid[bin]
                sigma = sigma_vs_wavelength[bin]
                if sigma > 0:
                    psf = np.exp(-0.5*((wavelengthGrid[bin-nhalf:bin+nhalf]-lam)/sigma)**2)
                    psf /= np.sum(psf)
                    rng = slice(nextIndex,nextIndex+psf.size)
                    sparseIndices[rng] = range(bin-nhalf,bin+nhalf)
                    sparseData[rng] = psf
                    nextIndex += psf.size
        sparseIndPtr[-1] = nextIndex
        resolution = sp.csc_matrix((sparseData,sparseIndices,sparseIndPtr),(nbins,nbins)).tocsr()
        return resolution

# resample camera throughput 
throughput_b=sim.WavelengthFunction(origin_wavelength,qsim.cameras[0].throughput).getResampledValues(bwaves)
throughput_r=sim.WavelengthFunction(origin_wavelength,qsim.cameras[1].throughput).getResampledValues(rwaves)
throughput_z=sim.WavelengthFunction(origin_wavelength,qsim.cameras[2].throughput).getResampledValues(zwaves)

#Sparse Matrices in three bands

resolution_b=resolution_matrix(sigma_b_vs_wave,bwaves,throughput_b).dot(np.identity(bmaxbin))
resolution_r=resolution_matrix(sigma_r_vs_wave,rwaves,throughput_r).dot(np.identity(rmaxbin))
resolution_z=resolution_matrix(sigma_z_vs_wave,zwaves,throughput_z).dot(np.identity(zmaxbin))

#print resolution_b.shape,resolution_r.shape,resolution_z.shape

# resolution data in format as desired for frame file
resolution_data=np.zeros((maxbin,3,21,500))
diags=np.arange(10,-11,-1)

for j,i in enumerate(diags):
    if i < 0:
       resolution_data[:i+bmaxbin-maxbin,0,j,args.nstart]=np.diagonal(resolution_b,offset=i)
       resolution_data[:i+rmaxbin-maxbin,1,j,args.nstart]=np.diagonal(resolution_r,offset=i)
       resolution_data[:i+zmaxbin-maxbin,2,j,args.nstart]=np.diagonal(resolution_z,offset=i)

    else:
       resolution_data[i:bmaxbin,0,j,args.nstart]=np.diagonal(resolution_b,offset=i)
       resolution_data[i:rmaxbin,1,j,args.nstart]=np.diagonal(resolution_r,offset=i)
       resolution_data[i:zmaxbin,2,j,args.nstart]=np.diagonal(resolution_z,offset=i)


# Now repeat the simulation for all spectra
 
print
for i in xrange(args.nstart,min(args.nspectra+args.nstart,objtype.shape[0]-args.nstart)): # Exclusive
    print "\rSimulating spectrum %d,  object type=%s"%(i,objtype[i]),
    sys.stdout.flush()
    specObj=sim.SpectralFluxDensity(wavelengths,spectra[i,:])
    results=qsim.simulate(sourceType=objtype[i].lower(),sourceSpectrum=specObj,airmass=args.airmass,expTime=args.exptime)
    nobj[:bmaxbin,0,i]=results.nobj[brange,0]
    nobj[:rmaxbin,1,i]=results.nobj[rrange,1]
    nobj[:zmaxbin,2,i]=results.nobj[zrange,2]

    nsky[:bmaxbin,0,i]=results.nsky[brange,0]
    nsky[:rmaxbin,1,i]=results.nsky[rrange,1]
    nsky[:zmaxbin,2,i]=results.nsky[zrange,2]

    nivar[:bmaxbin,0,i]=1/((results.nobj[brange,0])+(results.nsky[brange,0])+(results.rdnoise[brange,0])**2.0+(results.dknoise[brange,0])**2.0)
    nivar[:rmaxbin,1,i]=1/((results.nobj[rrange,1])+(results.nsky[rrange,1])+(results.rdnoise[rrange,1])**2.0+(results.dknoise[rrange,1])**2.0)
    nivar[:zmaxbin,2,i]=1/((results.nobj[zrange,2])+(results.nsky[zrange,2])+(results.rdnoise[zrange,2])**2.0+(results.dknoise[zrange,2])**2.0)


    sky_ivar[:bmaxbin,0,i]=25*1/((results.nsky[brange,0])+(results.rdnoise[brange,0])**2.0+(results.dknoise[brange,0])**2.0)
    sky_ivar[:rmaxbin,1,i]=25*1/((results.nsky[rrange,1])+(results.rdnoise[rrange,1])**2.0+(results.dknoise[rrange,1])**2.0)
    sky_ivar[:zmaxbin,2,i]=25*1/((results.nsky[zrange,2])+(results.rdnoise[zrange,2])**2.0+(results.dknoise[zrange,2])**2.0)


    cframe_observedflux[:bmaxbin,0,i]=results.obsflux[brange]
    cframe_observedflux[:rmaxbin,1,i]=results.obsflux[rrange]
    cframe_observedflux[:zmaxbin,2,i]=results.obsflux[zrange]

    cframe_ivar[:bmaxbin,0,i]=results.ivar[brange]
    cframe_ivar[:rmaxbin,1,i]=results.ivar[rrange]
    cframe_ivar[:zmaxbin,2,i]=results.ivar[zrange]
	
    rand_noise[:bmaxbin,0,i]=np.random.normal(np.zeros(bmaxbin),np.ones(bmaxbin)/np.sqrt(nivar[:bmaxbin,0,i]),bmaxbin)
    rand_noise[:rmaxbin,1,i]=np.random.normal(np.zeros(rmaxbin),np.ones(rmaxbin)/np.sqrt(nivar[:rmaxbin,1,i]),rmaxbin)
    rand_noise[:zmaxbin,2,i]=np.random.normal(np.zeros(zmaxbin),np.ones(zmaxbin)/np.sqrt(nivar[:zmaxbin,2,i]),zmaxbin)

    sky_rand_noise[:bmaxbin,0,i]=np.random.normal(np.zeros(bmaxbin),np.ones(bmaxbin)/np.sqrt(sky_ivar[:bmaxbin,0,i]),bmaxbin)
    sky_rand_noise[:rmaxbin,1,i]=np.random.normal(np.zeros(rmaxbin),np.ones(rmaxbin)/np.sqrt(sky_ivar[:rmaxbin,1,i]),rmaxbin)
    sky_rand_noise[:zmaxbin,2,i]=np.random.normal(np.zeros(zmaxbin),np.ones(zmaxbin)/np.sqrt(sky_ivar[:zmaxbin,2,i]),zmaxbin)


#resolution 
 
    # first resample the sigma_vs_wavelength for each fiber
    sigma_b_vs_wave=sim.WavelengthFunction(origin_wavelength,qsim.cameras[0].sigma_wave).getResampledValues(bwaves)
    sigma_r_vs_wave=sim.WavelengthFunction(origin_wavelength,qsim.cameras[1].sigma_wave).getResampledValues(rwaves)
    sigma_z_vs_wave=sim.WavelengthFunction(origin_wavelength,qsim.cameras[2].sigma_wave).getResampledValues(zwaves)
 
    #Sparse Matrices
    #- TODO: this part is slow
    resolution_b=resolution_matrix(sigma_b_vs_wave,bwaves,throughput_b).dot(np.identity(bmaxbin))
    resolution_r=resolution_matrix(sigma_r_vs_wave,rwaves,throughput_r).dot(np.identity(rmaxbin))
    resolution_z=resolution_matrix(sigma_z_vs_wave,zwaves,throughput_z).dot(np.identity(zmaxbin))
 
    #data 
    for j,k in enumerate(diags):
        if k < 0:
            resolution_data[:k+bmaxbin-maxbin,0,j,i]=np.diagonal(resolution_b,offset=k)
            resolution_data[:k+rmaxbin-maxbin,1,j,i]=np.diagonal(resolution_r,offset=k)
            resolution_data[:k+zmaxbin-maxbin,2,j,i]=np.diagonal(resolution_z,offset=k)

        else:
            resolution_data[k:bmaxbin,0,j,i]=np.diagonal(resolution_b,offset=k)
            resolution_data[k:rmaxbin,1,j,i]=np.diagonal(resolution_r,offset=k)
            resolution_data[k:zmaxbin,2,j,i]=np.diagonal(resolution_z,offset=k)



armName={"b":0,"r":1,"z":2}
armWaves={"b":bwaves,"r":rwaves,"z":zwaves}
armBins={"b":bmaxbin,"r":rmaxbin,"z":zmaxbin}


#armResolution={"b":resolution[:,0,:,:],"r":resolution[:,1,:,:],"z":resolution[:,2,:,:]}

#Need Four Files to write: May need to configure which ones to output, rather than all. 
#1. frame file: (x3)
#2. skymodel file:(x3)
#3. flux calibration vector file (x3)
#4. cframe file

#All files will be written here:
# Need to sync with write in desispec.io ? But there are few issues to be accounted.

#filePath=EXPID_DIR+'/'

def do_convolve(wave,resolution,flux):
    nwave=len(wave)
    R=Resolution(resolution)
    convolved=R.dot(flux)
    return convolved


for arm in ["b","r","z"]:	

############--------------------------------------------------------- 
    ###frame file 

    framefileName=desispec.io.findfile("frame",NIGHT,EXPID,"%s%s"%(arm,spectrograph))
    frame_flux=np.transpose(nobj[:armBins[arm],armName[arm],args.nstart:args.nstart+args.nspectra]+nsky[:armBins[arm],armName[arm],args.nstart:args.nstart+args.nspectra]+rand_noise[:armBins[arm],armName[arm],args.nstart:args.nstart+args.nspectra])
    frame_ivar=np.transpose(nivar[:armBins[arm],armName[arm],args.nstart:args.nstart+args.nspectra])
    resol=np.transpose(resolution_data[:armBins[arm],armName[arm],:,args.nstart:args.nstart+args.nspectra])
    # write frame file
    desispec.io.frame.write_frame(framefileName,frame_flux,frame_ivar,armWaves[arm],resol,header=None)

############--------------------------------------------------------
    #cframe file
    
    cframeFileName=desispec.io.findfile("cframe",NIGHT,EXPID,"%s%s"%(arm,spectrograph))
    cframeFlux=np.transpose(cframe_observedflux[:armBins[arm],armName[arm],args.nstart:args.nstart+args.nspectra])
    cframeIvar=np.transpose(cframe_ivar[:armBins[arm],armName[arm],args.nstart:args.nstart+args.nspectra])
    
    # write cframe file
    desispec.io.frame.write_frame(cframeFileName,cframeFlux,cframeIvar,armWaves[arm],resol,header=None)

############-----------------------------------------------------
    #sky file (for now taking only 1D, should change to (nspec,nwave) format)
    
    skyfileName=desispec.io.findfile("sky",NIGHT,EXPID,"%s%s"%(arm,spectrograph))
    skyflux=np.transpose(nsky[:armBins[arm],armName[arm],10]+sky_rand_noise[:armBins[arm],armName[arm],10])
    skyivar=np.transpose(sky_ivar[:armBins[arm],armName[arm],10])
    skymask=np.zeros(skyflux.shape, dtype=int)
    cskyflux=do_convolve(armWaves[arm],np.transpose(resolution_data[:armBins[arm],armName[arm],:,10]),skyflux)
    cskyivar=do_convolve(armWaves[arm],np.transpose(resolution_data[:armBins[arm],armName[arm],:,10]),skyivar)
    
    #write sky file 
    desispec.io.sky.write_sky(skyfileName,skyflux,skyivar,skymask,cskyflux,cskyivar,armWaves[arm],header=None)

############----------------------------------------------------------
    #calibration vector file

    calibVectorFile=desispec.io.findfile("calib",NIGHT,EXPID,"%s%s"%(arm,spectrograph))
    calibration=np.transpose(cframe_observedflux[:armBins[arm],armName[arm],args.nstart:args.nstart+args.nspectra]/nobj[:armBins[arm],armName[arm],args.nstart:args.nstart+args.nspectra])
    # Gives RuntimeWarning: invalid value encountered in divide. Correct?

    calibivar=np.transpose(cframe_ivar[:armBins[arm],armName[arm],args.nstart:args.nstart+args.nspectra])*calibration
    #mask=(1/calibivar>0).astype(long)??
    mask=np.zeros(calibration.shape, dtype=int)
    n_spec=calibration.shape[0]
    n_wave=calibration.shape[1]
    ccalibration=np.zeros((n_spec,n_wave))
    ccalibivar=np.zeros((n_spec,n_wave))
    for i in range(n_spec):
        ccalibration[i,:]=do_convolve(armWaves[arm],np.transpose(resolution_data[:armBins[arm],armName[arm],:,i]),calibration[i,:])
        ccalibivar[i,:]=do_convolve(armWaves[arm],np.transpose(resolution_data[:armBins[arm],armName[arm],:,i]),calibivar[i,:])
    #header from the frame file??
    head=pyfits.getheader(framefileName)
    #print head
    # write result
    write_flux_calibration(calibVectorFile,calibration, calibivar, mask, ccalibration, ccalibivar,armWaves[arm],head)
    
filePath=os.path.join(prod_Dir,'exposures',NIGHT,"%08d"%EXPID)
print "Wrote files to", filePath
 
#spectrograph=spectrograph+1	
