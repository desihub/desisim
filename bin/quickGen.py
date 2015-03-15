#!/usr/bin/env python

#*******************************
# A quick output of quicksim simulation for DESI
# For each 500 input spectra, this wrapper Outputs Four files for each arm B,R,Z,
#	1. frame file (x3)
#	2. Flux Calibration Vector file (x3)
#	3. Sky Model File (x3)
#	4. cframe file (x3)
# One full exposure will have 12*10=120 files
# version 0.1
# Govinda Dhungana/ Sami Kama, Jan 2015 
# gdhungan_at_smu_dot_edu
#
#Example:
#
#********************************

import argparse
import os
import os.path
import numpy as np
import astropy.io.fits as pyfits
import sys
import desimodel.simulate as sim


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

#But read fibermap file and extract the headers needed for Directory tree

#read fibermapfile to get objecttype,NIGHT and EXPID....
if args.fiberfile:
    print "Opening fiber file %s"%(args.fiberfile)
    fiber_hdulist=pyfits.open(args.fiberfile)
    objtype=fiber_hdulist[1].data['OBJTYPE'].copy()
    #need to replace STD object types with STAR since quicksim expects star instead of std
    stdindx=np.where(objtype=='STD') # match STD with STAR
    objtype[stdindx]='STAR'
    #print objtype[0],type(objtype[0]),type(objtype),objtype.shape
    NIGHT=fiber_hdulist[1].header['NIGHT']
    EXPID=fiber_hdulist[1].header['EXPID']
    fiber_hdulist.close()
else:
    fiber_hdulist=None


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

exposure_Dir=os.path.join(prod_Dir,'exposures')
if os.path.exists(exposure_Dir):
    if not os.path.isdir(exposure_Dir):
        raise RuntimeError("Path %s Not a directory"%exposure_DIR)
else:
    try:
        os.makedirs(exposure_Dir)
    except:
        raise

#----------NIGHT--------------

NIGHT_DIR=os.path.join(exposure_Dir,NIGHT)
if os.path.exists(NIGHT_DIR):
    if not os.path.isdir(NIGHT_DIR):
        raise RuntimeError("Path %s Not a directory"%NIGHT_DIR)
else:
    try:
        os.makedirs(NIGHT_DIR)
    except:
        raise

#---------EXPID-----------

EXPID_DIR=os.path.join(NIGHT_DIR,"%09d"%EXPID)
if os.path.exists(EXPID_DIR):
    if not os.path.isdir(EXPID_DIR):
        raise RuntimeError("Path %s Not a directory"%EXPID_DIR)
else:
    try:
        os.makedirs(EXPID_DIR)
    except:
        raise

if args.input is None:
    print('sys.stderr,"ERROR -i/--input filename required"')



#read the input file (simspec file)
print 'Now Reading the input file',args.input

hdulist=pyfits.open(args.input)
data=hdulist[0].data
hdr=hdulist[0].header
wavelengths=hdr["CRVAL1"]+hdr["CDELT1"]*np.arange(len(data[1,:]))
spectra=data/1.0e-17# flux in units of 1.0e-17 ergs/cm^2/s/A

#print "File Shape:", data.shape
print "wavelength range:", wavelengths[0], "to", wavelengths[-1]
nspec=data.shape[0]
nwave=data.shape[1]

# Here we will run for single CCD(500 x 3 in total). Fewer spectra can be run using 'nspectra' and 'nstart' options

spectrograph=0

print " simulating spectra",args.nstart, "to", args.nspectra+args.nstart-1

print "************************************************"


# Run Simulation for first Spectrum
print "Initializing QuickSim"
specObj0=sim.SpectralFluxDensity(wavelengths,spectra[args.nstart,:])
qsim=sim.Quick(basePath=DESIMODEL_DIR)
print "Simulating Spectrum",args.nstart," of spectrograph",spectrograph, "object type:", objtype[args.nstart]

# Why athmosphere is not included?
results=qsim.simulate(sourceType=objtype[args.nstart].lower(),sourceSpectrum=specObj0,airmass=args.airmass,expTime=args.exptime)
observedWavelengths=results.wave


# Define wavelimits for B,R,Z camera(by hand this time), later simulated data is cropped within these limits

waveLimits={'b':(3569,5949),'r':(5625,7741),'z':(7435,9834)}
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

# Now repeat the simulation for all spectra
 
for i in xrange(args.nstart+1,min(args.nspectra+args.nstart,objtype.shape[0]-args.nstart)): # Exclusive
    print "\rSimulating %d object type=%s"%(i,objtype[i]),
    sys.stdout.flush()
    specObj=sim.SpectralFluxDensity(wavelengths,spectra[i,:])
    results=qsim.simulate(sourceType=objtype[i].lower(),sourceSpectrum=specObj,airmass=args.airmass,expTime=args.exptime)
    #print results.nobj.shape,results.nsky.shape,results.wave.shape
	
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

    cframe_ivar[:bmaxbin,0,i]=results.ivar[brange]
    cframe_ivar[:rmaxbin,1,i]=results.ivar[rrange]
    cframe_ivar[:zmaxbin,2,i]=results.ivar[zrange]
    cframe_observedflux[:bmaxbin,0,i]=results.obsflux[brange]
    cframe_observedflux[:rmaxbin,1,i]=results.obsflux[rrange]
    cframe_observedflux[:zmaxbin,2,i]=results.obsflux[zrange]
	
    rand_noise[:bmaxbin,0,i]=np.random.normal(np.zeros(bmaxbin),np.ones(bmaxbin)/np.sqrt(nivar[:bmaxbin,0,i]),bmaxbin)
    rand_noise[:rmaxbin,1,i]=np.random.normal(np.zeros(rmaxbin),np.ones(rmaxbin)/np.sqrt(nivar[:rmaxbin,1,i]),rmaxbin)
    rand_noise[:zmaxbin,2,i]=np.random.normal(np.zeros(zmaxbin),np.ones(zmaxbin)/np.sqrt(nivar[:zmaxbin,2,i]),zmaxbin)

    sky_rand_noise[:bmaxbin,0,i]=np.random.normal(np.zeros(bmaxbin),np.ones(bmaxbin)/np.sqrt(sky_ivar[:bmaxbin,0,i]),bmaxbin)
    sky_rand_noise[:rmaxbin,1,i]=np.random.normal(np.zeros(rmaxbin),np.ones(rmaxbin)/np.sqrt(sky_ivar[:rmaxbin,1,i]),rmaxbin)
    sky_rand_noise[:zmaxbin,2,i]=np.random.normal(np.zeros(zmaxbin),np.ones(zmaxbin)/np.sqrt(sky_ivar[:zmaxbin,2,i]),zmaxbin)

print

armName={"b":0,"r":1,"z":2}
armWaves={"b":bwaves,"r":rwaves,"z":zwaves}
armBins={"b":bmaxbin,"r":rmaxbin,"z":zmaxbin}

#Need Four Files to write: May need to configure which ones to output, rather than all. 
#1. frame file: (x3)
#2. skymodel file:(x3)
#3. flux calibration vector file (x3)
#4. cframe file

#All files will be written here:
filePath=EXPID_DIR+'/'

for arm in ["b","r","z"]:	

############----------frame file------------------ 

    framefileName="frame-%s%s-%09d.fits"%(arm,spectrograph,EXPID)

    PrimaryImage=pyfits.PrimaryHDU(np.transpose(nobj[:armBins[arm],armName[arm],:]+nsky[:armBins[arm],armName[arm],:]+rand_noise[:armBins[arm],armName[arm],:])) # This is object+sky photon counts + Random Noise (from nivar)

    print "Shapes ",framefileName,(armBins[arm],args.nspectra)
    nivarImage=pyfits.ImageHDU(data=np.transpose(nivar[:armBins[arm],armName[arm],:]),name="IVAR")
    WaveImage=pyfits.ImageHDU(data=armWaves[arm],name="WAVELENGTH")

    #HDU0- object+sky Counts

    PrimaryImage.header["NAXIS1"]=(armBins[arm],"Number of wavelength samples")
    PrimaryImage.header["NAXIS2"]=(args.nspectra,"Number of extracted spectra")
    PrimaryImage.header["EXTNAME"]="FLUX"
    PrimaryImage.header["CRVAL1"]=(waveLimits[arm][0],"Starting wavelength [Angstroms]")
    PrimaryImage.header["CDELT1"]=(hdr["CDELT1"],"Wavelength step [Angstroms]")
    PrimaryImage.header["AIRORVAC"]=("vac","Vacuum wavelengths")
    PrimaryImage.header["LOGLAM"]=(0,"linear wavelength steps, not log10")
    PrimaryImage.header["SIMFILE"]=(args.input,"Input simulation file")
    PrimaryImage.header["CAMERA"]=(arm+str(spectrograph),"Spectograph Camera")
    PrimaryImage.header["VSPECTER"]=0
    PrimaryImage.header["EXPTIME"]=(args.exptime,"Exposure time [sec]")
    PrimaryImage.header["RDNOISE"]=(0,"Read noise [electrons]")
    PrimaryImage.header["FLAVOR"]=("science","Exposure type (arc, flat, science)")
    PrimaryImage.header["SPECMIN"]=(0,"First spectrum")
    PrimaryImage.header["SPECMAX"]=(min(args.nspectra,objtype.shape[0])-1,"Last spectrum")
    PrimaryImage.header["NSPEC"]=(min(args.nspectra,objtype.shape[0]),"Number of spectra") 
    PrimaryImage.header["WAVEMIN"]=(min(armWaves[arm]),"First wavelength [Angstroms]")
    PrimaryImage.header["WAVEMAX"]=(max(armWaves[arm]),"Last wavelength [Angstroms]")
    PrimaryImage.header["WAVESTEP"]=(np.gradient(armWaves[arm])[0],"Wavelength step size [Angstroms]")
    PrimaryImage.header["SPECTER"]=0
    #PrimaryImage.header["IN_PSF"]=(".../desimodel/trunk/data/specpsf/psf-"+arm+".fits","Input spectral PSF")
    #PrimaryImage.header["IN_IMG"]=("...im/alpha-3/20150107/pix-"+arm+"0-"+EXPID+".fits","Input image")

    # HDU1 - Inverse Variance (on Object+Sky)

    nivarImage.header["NAXIS1"]=(armBins[arm],"Number of wavlengths")
    nivarImage.header["NAXIS2"]=(args.nspectra,"Number of spectra")
    nivarImage.header["EXTNAME"]="IVAR"

    # HDU2 - Wavelength

    WaveImage.header["NAXIS1"]=(armBins[arm],"Number of wavelengths")
    WaveImage.header["EXTNAME"]="WAVELENGTH"

    framehdulist=pyfits.HDUList([PrimaryImage,nivarImage,WaveImage])
    prihdr=framehdulist[0].header
    framehdulist.writeto(filePath+framefileName,clobber=True)
    framehdulist.close()

#######------------------skymodel file--------------------------

    skyfileName="skymodel-%s%s-%09d.fits"%(arm,spectrograph,EXPID)
    skyImage=pyfits.PrimaryHDU(np.transpose(nsky[:armBins[arm],armName[arm],:]+sky_rand_noise[:armBins[arm],armName[arm],:])) # SKY counts+ Random Noise
    skyIvar=pyfits.ImageHDU(data=np.transpose(sky_ivar[:armBins[arm],armName[arm],:]),name="IVAR")

	#HDU0 - Sky Photons	
    skyImage.header["NAXIS1"]=armBins[arm]
    skyImage.header["NAXIS2"]=args.nspectra
    skyImage.header["CRVAL1"]=(waveLimits[arm][0],"Starting wavelength [Angstroms]")
    skyImage.header["CDELT1"]=(np.gradient(armWaves[arm])[0],"Wavelength step size [Angstroms]")
    skyImage.header["EXTNAME"]='SKY_PHOTONS' #What to call this?

    #HDU1 - Inverse Variance(sky)
    skyIvar.header["NAXIS"]=armBins[arm]
    skyIvar.header["EXTNAME"]='IVAR'
	
    skyhdulist=pyfits.HDUList([skyImage,skyIvar])
    prihdr=skyhdulist[0].header
    skyhdulist.writeto(filePath+skyfileName,clobber=True)
    skyhdulist.close()

######-------------------------cframe file---------------------------


    cframeFileName="cframe-%s%s-%09d.fits"%(arm,spectrograph,EXPID)
    cframeImage=pyfits.PrimaryHDU(np.transpose(cframe_observedflux[:armBins[arm],armName[arm],:]))
    fluxIvarImage=pyfits.ImageHDU(data=np.transpose(cframe_ivar[:armBins[arm],armName[arm],:]),name="IVAR")
    #maskImage=pyfits.ImageHDU(data=???,name="MASK")

    #HDU0 - Calibrated Flux ( erg/s/cm2/A)
	
    cframeImage.header["NAXIS1"]=armBins[arm]
    cframeImage.header["CRVAL1"]=waveLimits[arm][0]
    cframeImage.header["EXTNAME"]=("FLUX","erg/s/cm2/A")
	
    #HDU1 - Inverse Variance in Flux (erg/s/cm2/A)^-2
    fluxIvarImage.header["NAXIS"]=0
    fluxIvarImage.header["EXTNAME"]=('IVAR',"(erg/s/cm2/A)^-2")
	
    cframehduList=pyfits.HDUList([cframeImage,fluxIvarImage])
    prihdr=cframehduList[0].header
    cframehduList.writeto(filePath+cframeFileName,clobber=True)
    cframehduList.close()

########--------------------calibration vector file-----------------
	
    calibVectorFile="fluxcalib-%s%s-%09d.fits"%(arm,spectrograph,EXPID)
    calibImage=pyfits.PrimaryHDU(np.transpose(cframe_observedflux[:armBins[arm],armName[arm],:]/nobj[:armBins[arm],armName[arm],:]))

    #calibMetadata=pyfits.ImageHDU(???)

    #HDU0-Calibration Vector
    calibImage.header["NAXIS1"]=armBins[arm]
    calibImage.header["NAXIS2"]=args.nspectra
    calibImage.header["CRVAL1"]=0
    calibImage.header["CDELT1"]=0
    calibImage.header["EXTNAME"]='CALIB'

    #HDU1 - Metadata(???)- Details to go here!!!
	
    calibhdulist=pyfits.HDUList([calibImage])
    prihdr=calibhdulist[0].header
    calibhdulist.writeto(filePath+calibVectorFile,clobber=True)
    calibhdulist.close()
	
<<<<<<< HEAD
print "Wrote files to", filePath
 
=======
print "Wrote files to", filePath	

>>>>>>> upstream/master
#spectrograph=spectrograph+1	
		

