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
import scipy.special
import astropy.io.fits as pyfits
import sys
import desimodel.simulate as sim
import desispec
import desisim.io
from desispec.resolution import Resolution
from desispec.io import write_flux_calibration, write_fiberflat
from desispec.interpolation import resample_flux
from desispec.frame import Frame
from desispec.fiberflat import FiberFlat
from desispec.sky import SkyModel
from desispec.fluxcalibration import FluxCalib
import matplotlib.pyplot as plt

#- Parse arguments
parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#      usage = "%prog [arguments]"
#      )


parser.add_argument("--input",type=str, help="input spectra")
parser.add_argument("--fiberfile",type=str, help='fiber map file')
### parser.add_argument("--exptime", type=int, help="exposure time")
parser.add_argument("--nspectra",type=int,default=500,help='no. of spectra to be simulated, starting from first')
parser.add_argument("--nstart", type=int, default=0,help='starting spectra # for simulation')
parser.add_argument("--spectrograph",type=int, default=0,help='Spectrograph no. 0-9')
### parser.add_argument("--airmass",type=float, help="airmass")
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
    tbdata,hdr=desispec.io.fibermap.read_fibermap(args.fiberfile, header=True)
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

#- TODO: make this an option
#spectrograph=0

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
results=qsim.simulate(sourceType='star', sourceSpectrum=specObj0,
    airmass=simspec.header['AIRMASS'], expTime=simspec.header['EXPTIME'])
observedWavelengths=results.wave
origin_wavelength=qsim.wavelengthGrid
waveLimits={'b':(3569,5949),'r':(5625,7741),'z':(7435,9834)}

#- Check if input simspec is for a continuum flat lamp instead of science
if simspec.flavor == 'flat':
    print "Simulating flat lamp exposure"
    for channel in ('b', 'r', 'z'):
        camera = channel+str(args.spectrograph)
        outfile = desispec.io.findfile('fiberflat', NIGHT, EXPID, camera)
        ii = (waveLimits[channel][0] <= observedWavelengths) & (observedWavelengths <= waveLimits[channel][1])
        wave = observedWavelengths[ii]
        dw = np.gradient(simspec.wave[channel])
        meanspec = resample_flux(wave, simspec.wave[channel], np.average(simspec.phot[channel]/dw, axis=0))
        fiberflat = np.random.normal(loc=0.0, scale=1.0/np.sqrt(meanspec), size=(nspec, len(wave)))
        ivar = np.tile(1.0/meanspec, nspec).reshape(nspec, len(meanspec))
        mask = np.zeros((simspec.nspec, len(wave)))
        
        ff = FiberFlat(wave, fiberflat, ivar, mask, meanspec)        
        write_fiberflat(outfile, ff)
        print "Wrote "+outfile
    
    sys.exit(0)    

# Define wavelimits for B,R,Z camera(by hand this time), later simulated data is cropped within these limits

# Get the camera ranges from quicksim outputs and save the outputs for each camera
waveRange=dict()
waveMaxbin=dict()
waves=dict()
for i,channel in enumerate(['b','r','z']):
    waveRange[channel],=np.where((observedWavelengths>=waveLimits[channel][0])&(observedWavelengths<=waveLimits[channel][1]))
    waveMaxbin[channel],=waveRange[channel].shape
    waves[channel]=observedWavelengths[waveRange[channel]]

maxbin=max(waveMaxbin['b'],waveMaxbin['r'],waveMaxbin['z'])

# Now break the simulated outputs in three different ranges. 

nobj=np.zeros((500,3,maxbin))     # Object Photons
nsky=np.zeros((500,3,maxbin))      # sky photons
nivar=np.zeros((500,3,maxbin))     # inverse variance  (object+sky)
sky_ivar=np.zeros((500,3,maxbin)) # inverse variance of sky
cframe_observedflux=np.zeros((500,3,maxbin))    # calibrated object flux
cframe_ivar=np.zeros((500,3,maxbin))    # inverse variance of calibrated object flux
frame_rand_noise=np.zeros((500,3,maxbin))     # random Gaussian noise to nobj+nsky
sky_rand_noise=np.zeros((500,3,maxbin))  # random Gaussian noise to sky only
cframe_rand_noise=np.zeros((500,3,maxbin))  # random Gaussian noise to calibrated flux

# Now initial values
np.random.seed(0)

#construct resolution matrix from sigma_vs_wavelength. First resample to respective sigmas vs wavelengths

sigma_vs_wave = dict()
for i, channel in enumerate( ['b', 'r', 'z'] ):
    sigma_vs_wave[channel] = sim.WavelengthFunction(origin_wavelength, qsim.cameras[i].sigma_wave).getResampledValues(waves[channel])

#- Resolution data from gaussian sigmas
def _calc_resolution_data(sigma, wavelengths, nspec):
    """
    Return resolution matrix data
    
    Args:
        sigma : 1D array of sigmas in Angstroms
        wavelengths : 1D array of equally spaced wavelengths in Angstroms
        
    TODO: check this.  Replace with desisim.resolution.Resolution.
    """
    ndiag = 21
    offsets = np.arange(-ndiag//2+1,ndiag//2+1,1.0)
    nwave = len(wavelengths)
    dw = np.gradient(wavelengths)
    
    resolution_data = np.zeros((ndiag, nwave))
    for i in range(nwave):
        x = offsets * dw[i] / sigma[i]
        dx = dw[i] / sigma[i]

        edges = np.concatenate([x-dx/2, x[-1:]+dx/2])
        assert len(edges) == len(x)+1

        y = scipy.special.erf(edges)
        resolution_data[:, i] = (y[1:] - y[:-1])/2
        
    #- Convert this to [nspec, ndiag, nwave]
    result = np.zeros((nspec, ndiag, nwave))
    for i in range(nspec):
        result[i] = resolution_data
        
    return result

#-------------------------------------------------------------------------

# resolution data in format as desired for frame file
nspec=args.nspectra
resolution_data = dict()
for i, channel in enumerate( ['b', 'r', 'z'] ):
    resolution_data[channel] = _calc_resolution_data(sigma_vs_wave[channel], waves[channel], nspec)


# Now repeat the simulation for all spectra
 

for j in xrange(args.nstart,min(args.nspectra+args.nstart,objtype.shape[0]-args.nstart)): # Exclusive
    print "\rSimulating spectrum %d,  object type=%s"%(j,objtype[j]),
    sys.stdout.flush()
    specObj=sim.SpectralFluxDensity(wavelengths,spectra[j,:])
    results=qsim.simulate(sourceType=objtype[j].lower(),sourceSpectrum=specObj,
            airmass=simspec.header['AIRMASS'], expTime=simspec.header['EXPTIME'])
    for i,channel in enumerate(['b','r','z']):
        nobj[j,i,:waveMaxbin[channel]]=results.nobj[waveRange[channel],i]

        nsky[j,i,:waveMaxbin[channel]]=results.nsky[waveRange[channel],i]

        nivar[j,i,:waveMaxbin[channel]]=1/((results.nobj[waveRange[channel],i])+(results.nsky[waveRange[channel],i])+(results.rdnoise[waveRange[channel],i])**2.0+(results.dknoise[waveRange[channel],i])**2.0)

        sky_ivar[j,i,:waveMaxbin[channel]]=25*1/((results.nsky[waveRange[channel],i])+(results.rdnoise[waveRange[channel],i])**2.0+(results.dknoise[waveRange[channel],i])**2.0)

        cframe_observedflux[j,i,:waveMaxbin[channel]]=results.obsflux[waveRange[channel]]*1.0e-17
        cframe_ivar[j,i,:waveMaxbin[channel]]=np.clip(results.ivar[waveRange[channel]]*1.0e34, 1e-12, np.max(results.ivar[waveRange[channel]]))

        frame_rand_noise[j,i,:waveMaxbin[channel]]=np.random.normal(np.zeros(waveMaxbin[channel]),np.ones(waveMaxbin[channel])/np.sqrt(nivar[j,i,:waveMaxbin[channel]]),waveMaxbin[channel])

        sky_rand_noise[j,i,:waveMaxbin[channel]]=np.random.normal(np.zeros(waveMaxbin[channel]),np.ones(waveMaxbin[channel])/np.sqrt(sky_ivar[j,i,:waveMaxbin[channel]]),waveMaxbin[channel])

        cframe_rand_noise[j,i,:waveMaxbin[channel]]=np.random.normal(0.0,1.0/np.sqrt(cframe_ivar[j,i,:waveMaxbin[channel]]))


print
armName={"b":0,"r":1,"z":2}

#Need Four Files to write: May need to configure which ones to output, rather than all. 
#1. frame file: (x3)
#2. skymodel file:(x3)
#3. flux calibration vector file (x3)
#4. cframe file

def do_convolve(wave,resolution,flux):
    """
    Returns the convolved flux
    Args:
        wave : wavelength for the given channel
        resolution : resolution data for single fiber
        flux: single fiber flux
    """
    R=Resolution(resolution)
    convolved=R.dot(flux)
    return convolved

for channel in ["b","r","z"]:	

    #Before writing, convert from counts/bin to counts/A (as in Pixsim output)
    #Quicksim Default:
    #FLUX - input spectrum resampled to this binning; no noise added [1e-17 erg/s/cm2/s/Ang] 
    #COUNTS_OBJ - object counts in 0.5 Ang bin
    #COUNTS_SKY - sky counts in 0.5 Ang bin
 
    dwave=np.gradient(waves[channel])
    nobj[:,armName[channel],:waveMaxbin[channel]]/=dwave
    frame_rand_noise[:,armName[channel],:waveMaxbin[channel]]/=dwave
    nivar[:,armName[channel],:waveMaxbin[channel]]*=dwave**2

    nsky[:,armName[channel],:waveMaxbin[channel]]/=dwave
    sky_rand_noise[:,armName[channel],:waveMaxbin[channel]]/=dwave
    sky_ivar[:,armName[channel],:waveMaxbin[channel]]/=dwave**2

    
############--------------------------------------------------------- 
    ###frame file 

    framefileName=desispec.io.findfile("frame",NIGHT,EXPID,"%s%s"%(channel,args.spectrograph))
    frame_flux=nobj[args.nstart:args.nstart+args.nspectra,armName[channel],:waveMaxbin[channel]]+nsky[args.nstart:args.nstart+args.nspectra,armName[channel],:waveMaxbin[channel]]+frame_rand_noise[args.nstart:args.nstart+args.nspectra,armName[channel],:waveMaxbin[channel]]
    frame_ivar=nivar[args.nstart:args.nstart+args.nspectra,armName[channel],:waveMaxbin[channel]]

    # write frame file
    frame = Frame(waves[channel], frame_flux, frame_ivar, resolution_data=resolution_data[channel],spectrograph=args.spectrograph)
    desispec.io.write_frame(framefileName, frame)

############--------------------------------------------------------
    #cframe file
    #TODO : for sky input spectrum (all zero), quicksim output flux is also zero and also ivar. Need to fix!! 
    
    cframeFileName=desispec.io.findfile("cframe",NIGHT,EXPID,"%s%s"%(channel,args.spectrograph))
    cframeFlux=cframe_observedflux[args.nstart:args.nstart+args.nspectra,armName[channel],:waveMaxbin[channel]]+cframe_rand_noise[args.nstart:args.nstart+args.nspectra,armName[channel],:waveMaxbin[channel]]
    cframeIvar=cframe_ivar[args.nstart:args.nstart+args.nspectra,armName[channel],:waveMaxbin[channel]]
    
    # write cframe file
    cframe = Frame(waves[channel], cframeFlux, cframeIvar, resolution_data=resolution_data[channel],spectrograph=args.spectrograph)
    desispec.io.frame.write_frame(cframeFileName,cframe)

############-----------------------------------------------------
    #sky file (for now taking only 1D, should change to (nspec,nwave) format)
    
    skyfileName=desispec.io.findfile("sky",NIGHT,EXPID,"%s%s"%(channel,args.spectrograph))
    skyflux=nsky[:,armName[channel],:waveMaxbin[channel]]+sky_rand_noise[:,armName[channel],:waveMaxbin[channel]]
    skyivar=sky_ivar[:,armName[channel],:waveMaxbin[channel]]
    skymask=np.zeros(skyflux.shape, dtype=int)
    
    # write sky file 
    skymodel = SkyModel(waves[channel], skyflux, skyivar, skymask)
    desispec.io.sky.write_sky(skyfileName, skymodel)

############----------------------------------------------------------
    #calibration vector file
    #- TODO: this will get refactored into 2D outputs

    calibVectorFile=desispec.io.findfile("calib",NIGHT,EXPID,"%s%s"%(channel,args.spectrograph))
    flux = cframe_observedflux[args.nstart:args.nstart+args.nspectra,armName[channel],:waveMaxbin[channel]]
    phot = nobj[args.nstart:args.nstart+args.nspectra,armName[channel],:waveMaxbin[channel]]
    calibration = np.zeros_like(phot)
    ii = (flux>0)
    calibration[ii] = phot[ii] / flux[ii]

    #- TODO: what should calibivar be?
    #- For now, model it as the noise of combining ~10 spectra
    calibivar=10/cframe_ivar[args.nstart:args.nstart+args.nspectra,armName[channel],:waveMaxbin[channel]]
    #mask=(1/calibivar>0).astype(long)??
    mask=np.zeros(calibration.shape, dtype=int)

    # write flux calibration
    fluxcalib = FluxCalib(waves[channel], calibration, calibivar, mask)
    write_flux_calibration(calibVectorFile, fluxcalib)
    
filePath=os.path.join(prod_Dir,'exposures',NIGHT,"%08d"%EXPID)
print "Wrote files to", filePath
 
#spectrograph=spectrograph+1	
