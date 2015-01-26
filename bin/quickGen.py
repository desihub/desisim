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
parser.add_argument("--model",type=str, help='object models: ELG, LRG, QSO, STD')

parser.add_argument("--fiberfile",type=str, help='fiber map file')
parser.add_argument("--exptime", type=int, help="exposure time")
parser.add_argument("--nspectra",type=int, help='no. of spectra to be simulated, starting from first')
parser.add_argument("--nstart", type=int, default=None,help='starting spectra # for simulation')
parser.add_argument("--airmass",type=float, help="airmass")
args = parser.parse_args()

#must import desimodel
if 'DESIMODEL' not in os.environ:
        raise RuntimeError('The environment variable DESIMODEL must be set.')
DESIMODEL_DIR=os.environ['DESIMODEL'] 

if args.input is None:
	print('sys.stderr,"ERROR -i/--input filename required"')

#read fibermapfile to get objecttype, may be more info if needed
if args.fiberfile:
	print "Opening fiber file %s"%(args.fiberfile)
	fiber_hdulist=pyfits.open(args.fiberfile)
	objtype=fiber_hdulist[1].data['OBJTYPE'].copy()
	#need to replace STD object types with STAR since quicksim expects star instead of std
	stdindx=np.where(objtype=='STD') # match STD with STAR
	objtype[stdindx]='STAR'
	#print objtype[0],type(objtype[0]),type(objtype),objtype.shape
	fiber_hdulist.close()
else:
	objtype=[args.model]
	fiber_hdulist=None

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

EXPID=expid=args.input.split("-")[-1].split(".")[0]
print EXPID

# Here we will run for single CCD(500 x 3 in total). Fewer spectra can be run using 'nspectra' and 'nstart' options

spectrograph=0

if args.nstart is None:
	args.nstart=0
#print " simulating from spectra: ", args.nstart
	
if args.nspectra is None:
	args.nspectra = 500

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

waveLimits={0:(3569,5949),1:(5625,7741),2:(7435,9834)}
# Get the camera ranges from quicksim outputs 
brange,=np.where((observedWavelengths>=waveLimits[0][0])&(observedWavelengths<=waveLimits[0][1]))
rrange,=np.where((observedWavelengths>=waveLimits[1][0])&(observedWavelengths<=waveLimits[1][1]))
zrange,=np.where((observedWavelengths>=waveLimits[2][0])&(observedWavelengths<=waveLimits[2][1]))

#number of points in each camera
armLimits={0:waveLimits[0],1:waveLimits[1],2:waveLimits[2]}
# save the observation wavelengths
bwaves=observedWavelengths[brange]
rwaves=observedWavelengths[rrange]
zwaves=observedWavelengths[zrange]
bmaxbin,=brange.shape
zmaxbin,=zrange.shape
rmaxbin,=rrange.shape
maxbin=max(bmaxbin,zmaxbin,rmaxbin)

print "Bins for B:", bmaxbin,"  ","R:", rmaxbin,"  ", "Z:", zmaxbin
print "Arm Limits", armLimits

#print "shapes ivar=",results.ivar.shape," nobj=", results.nobj.shape, "nsky=", results.nsky.shape, "rdnoise=", results.rdnoise.shape, " dknoise=", results.dknoise.shape," wave=",results.wave.shape,"srcflux=",results.srcflux.shape,"obsflux=",results.obsflux.shape


# Now break the simulated outputs in three different ranges.  

# Object Photons

NOBJ=np.zeros((maxbin,results.nobj.shape[1],500))
NOBJ[:bmaxbin,0,args.nstart]=results.nobj[brange,0]
NOBJ[:rmaxbin,1,args.nstart]=results.nobj[rrange,1]
NOBJ[:zmaxbin,2,args.nstart]=results.nobj[zrange,2]

# Sky Photons

NSKY=np.zeros((maxbin,3,500))
NSKY[:bmaxbin,0,args.nstart]=results.nsky[brange,0]
NSKY[:rmaxbin,1,args.nstart]=results.nsky[rrange,1]
NSKY[:zmaxbin,2,args.nstart]=results.nsky[zrange,2]

# Inverse Variance (Object+Sky Photons)

NIVAR=np.zeros((maxbin,3,500))
NIVAR[:bmaxbin,0,args.nstart]=1/((results.nobj[brange,0])+(results.nsky[brange,0])+(results.rdnoise[brange,0])**2.0+(results.dknoise[brange,0])**2.0)
NIVAR[:rmaxbin,1,args.nstart]=1/((results.nobj[rrange,1])+(results.nsky[rrange,1])+(results.rdnoise[rrange,1])**2.0+(results.dknoise[rrange,1])**2.0)
NIVAR[:zmaxbin,2,args.nstart]=1/((results.nobj[zrange,2])+(results.nsky[zrange,2])+(results.rdnoise[zrange,2])**2.0+(results.dknoise[zrange,2])**2.0)

# Inverse Variance in Sky counts

SKY_IVAR=np.zeros((maxbin,3,500)) # Multiply by 25, decreasing the error by a factor of 5.
SKY_IVAR[:bmaxbin,0,args.nstart]=25*1/((results.nsky[brange,0])+(results.rdnoise[brange,0])**2.0+(results.dknoise[brange,0])**2.0)
SKY_IVAR[:rmaxbin,1,args.nstart]=25*1/((results.nsky[rrange,1])+(results.rdnoise[rrange,1])**2.0+(results.dknoise[rrange,1])**2.0)
SKY_IVAR[:zmaxbin,2,args.nstart]=25*1/((results.nsky[zrange,2])+(results.rdnoise[zrange,2])**2.0+(results.dknoise[zrange,2])**2.0)

# Calibrated Object Flux ( in units of 10^-17 ergs/s/cm2/A)

CFRAME_OBSERVEDFLUX=np.zeros((maxbin,3,500))
CFRAME_OBSERVEDFLUX[:bmaxbin,0,args.nstart]=results.obsflux[brange]
CFRAME_OBSERVEDFLUX[:rmaxbin,1,args.nstart]=results.obsflux[rrange]
CFRAME_OBSERVEDFLUX[:zmaxbin,2,args.nstart]=results.obsflux[zrange]

# Inverse Variance in Object Flux (in units of [10^-17 ergs/s/cm2/A]^-2)

CFRAME_IVAR=np.zeros((maxbin,3,500))
CFRAME_IVAR[:bmaxbin,0,args.nstart]=results.ivar[brange]
CFRAME_IVAR[:rmaxbin,1,args.nstart]=results.ivar[rrange]
CFRAME_IVAR[:zmaxbin,2,args.nstart]=results.ivar[zrange]
	
# Random Gaussian Noise to NOBJ+NSKY

RAND_NOISE=np.zeros((maxbin,3,500))
np.random.seed(0)
RAND_NOISE[:bmaxbin,0,args.nstart]=np.random.normal(np.zeros(bmaxbin),np.ones(bmaxbin)/np.sqrt(NIVAR[:bmaxbin,0,args.nstart]),bmaxbin)
RAND_NOISE[:rmaxbin,1,args.nstart]=np.random.normal(np.zeros(rmaxbin),np.ones(rmaxbin)/np.sqrt(NIVAR[:rmaxbin,1,args.nstart]),rmaxbin)
RAND_NOISE[:zmaxbin,2,args.nstart]=np.random.normal(np.zeros(zmaxbin),np.ones(zmaxbin)/np.sqrt(NIVAR[:zmaxbin,2,args.nstart]),zmaxbin)

# Random Gaussian Noise to SKY only

SKY_RAND_NOISE=np.zeros((maxbin,3,500))
SKY_RAND_NOISE[:bmaxbin,0,args.nstart]=np.random.normal(np.zeros(bmaxbin),np.ones(bmaxbin)/np.sqrt(SKY_IVAR[:bmaxbin,0,args.nstart]),bmaxbin)
SKY_RAND_NOISE[:rmaxbin,1,args.nstart]=np.random.normal(np.zeros(rmaxbin),np.ones(rmaxbin)/np.sqrt(SKY_IVAR[:rmaxbin,1,args.nstart]),rmaxbin)
SKY_RAND_NOISE[:zmaxbin,2,args.nstart]=np.random.normal(np.zeros(zmaxbin),np.ones(zmaxbin)/np.sqrt(SKY_IVAR[:zmaxbin,2,args.nstart]),zmaxbin)

# Now repeat the simulation for all spectra
 
for i in xrange(args.nstart+1,min(args.nspectra+args.nstart,objtype.shape[0]-args.nstart)): # Exclusive
	print "Simulating %d object type=%s"%(i,objtype[i])
	specObj=sim.SpectralFluxDensity(wavelengths,spectra[i,:])
	results=qsim.simulate(sourceType=objtype[i].lower(),sourceSpectrum=specObj,airmass=args.airmass,expTime=args.exptime)
	#print results.nobj.shape,results.nsky.shape,results.wave.shape
	
	NOBJ[:bmaxbin,0,i]=results.nobj[brange,0]
	NOBJ[:rmaxbin,1,i]=results.nobj[rrange,1]
	NOBJ[:zmaxbin,2,i]=results.nobj[zrange,2]

	NSKY[:bmaxbin,0,i]=results.nsky[brange,0]
	NSKY[:rmaxbin,1,i]=results.nsky[rrange,1]
	NSKY[:zmaxbin,2,i]=results.nsky[zrange,2]

	NIVAR[:bmaxbin,0,i]=1/((results.nobj[brange,0])+(results.nsky[brange,0])+(results.rdnoise[brange,0])**2.0+(results.dknoise[brange,0])**2.0)
	NIVAR[:rmaxbin,1,i]=1/((results.nobj[rrange,1])+(results.nsky[rrange,1])+(results.rdnoise[rrange,1])**2.0+(results.dknoise[rrange,1])**2.0)
	NIVAR[:zmaxbin,2,i]=1/((results.nobj[zrange,2])+(results.nsky[zrange,2])+(results.rdnoise[zrange,2])**2.0+(results.dknoise[zrange,2])**2.0)


	SKY_IVAR[:bmaxbin,0,i]=25*1/((results.nsky[brange,0])+(results.rdnoise[brange,0])**2.0+(results.dknoise[brange,0])**2.0)
	SKY_IVAR[:rmaxbin,1,i]=25*1/((results.nsky[rrange,1])+(results.rdnoise[rrange,1])**2.0+(results.dknoise[rrange,1])**2.0)
	SKY_IVAR[:zmaxbin,2,i]=25*1/((results.nsky[zrange,2])+(results.rdnoise[zrange,2])**2.0+(results.dknoise[zrange,2])**2.0)

	CFRAME_IVAR[:bmaxbin,0,i]=results.ivar[brange]
	CFRAME_IVAR[:rmaxbin,1,i]=results.ivar[rrange]
	CFRAME_IVAR[:zmaxbin,2,i]=results.ivar[zrange]
	CFRAME_OBSERVEDFLUX[:bmaxbin,0,i]=results.obsflux[brange]
	CFRAME_OBSERVEDFLUX[:rmaxbin,1,i]=results.obsflux[rrange]
	CFRAME_OBSERVEDFLUX[:zmaxbin,2,i]=results.obsflux[zrange]

	np.random.seed(0)
	
	RAND_NOISE[:bmaxbin,0,i]=np.random.normal(np.zeros(bmaxbin),np.ones(bmaxbin)/np.sqrt(NIVAR[:bmaxbin,0,i]),bmaxbin)
	RAND_NOISE[:rmaxbin,1,i]=np.random.normal(np.zeros(rmaxbin),np.ones(rmaxbin)/np.sqrt(NIVAR[:rmaxbin,1,i]),rmaxbin)
	RAND_NOISE[:zmaxbin,2,i]=np.random.normal(np.zeros(zmaxbin),np.ones(zmaxbin)/np.sqrt(NIVAR[:zmaxbin,2,i]),zmaxbin)

	SKY_RAND_NOISE[:bmaxbin,0,i]=np.random.normal(np.zeros(bmaxbin),np.ones(bmaxbin)/np.sqrt(SKY_IVAR[:bmaxbin,0,i]),bmaxbin)
	SKY_RAND_NOISE[:rmaxbin,1,i]=np.random.normal(np.zeros(rmaxbin),np.ones(rmaxbin)/np.sqrt(SKY_IVAR[:rmaxbin,1,i]),rmaxbin)
	SKY_RAND_NOISE[:zmaxbin,2,i]=np.random.normal(np.zeros(zmaxbin),np.ones(zmaxbin)/np.sqrt(SKY_IVAR[:zmaxbin,2,i]),zmaxbin)


armName={0:"b",1:"r",2:"z"}
armWaves={0:bwaves,1:rwaves,2:zwaves}
armBins={0:bmaxbin,1:rmaxbin,2:zmaxbin}
armRange={0:brange,1:rrange,2:zrange}

print " Length of Armbins", armBins[0],armBins[1],armBins[2]
#print NOBJ[:armBins[0],0,:].shape

	#Need Four Files to write: May need to configure which ones to output, rather than all. 
	#1. frame file: (x3)
	#2. skymodel file:(x3)
	#3. flux calibration vector file (x3)
	#4. cframe file
for arm in range(0,3):	

############----------frame file------------------ 

	framefileName="frame-%s%s-%s.fits"%(armName[arm],spectrograph,EXPID)

	PrimaryImage=pyfits.PrimaryHDU(NOBJ[:armBins[arm],arm,:]+NSKY[:armBins[arm],arm,:]+RAND_NOISE[:armBins[arm],arm,:]) # This is object+sky photon counts + Random Noise (from NIVAR)

	print "Shapes ",framefileName,(armBins[arm],args.nspectra)
	NIVARImage=pyfits.ImageHDU(data=NIVAR[:armBins[arm],arm,:],name="IVAR")
	WaveImage=pyfits.ImageHDU(data=armWaves[arm],name="WAVELENGTH")

       	#HDU0- object+sky Counts

	PrimaryImage.header["NAXIS1"]=(armBins[arm],"Number of wavelength samples")
	PrimaryImage.header["NAXIS2"]=(args.nspectra,"Number of extracted spectra")
	PrimaryImage.header["EXTNAME"]="FLUX"
	PrimaryImage.header["CRVAL1"]=(armLimits[arm][0],"Starting wavelength [Angstroms]")
	PrimaryImage.header["CDELT1"]=(hdr["CDELT1"],"Wavelength step [Angstroms]")
	PrimaryImage.header["AIRORVAC"]=("vac","Vacuum wavelengths")
	PrimaryImage.header["LOGLAM"]=(0,"linear wavelength steps, not log10")
	PrimaryImage.header["SIMFILE"]=(args.input,"Input simulation file")
	PrimaryImage.header["CAMERA"]=(armName[arm]+str(spectrograph),"Spectograph Camera")
	PrimaryImage.header["VSPECTER"]=0
	PrimaryImage.header["EXPTIME"]=(args.exptime,"Exposure time [sec]")
	PrimaryImage.header["RDNOISE"]=(0,"Read noise [electrons]")#results.rdnoise[armRange[arm],arm]???
	PrimaryImage.header["FLAVOR"]=("science","Exposure type (arc, flat, science)")
	PrimaryImage.header["SPECMIN"]=(0,"First spectrum")
	PrimaryImage.header["SPECMAX"]=(min(args.nspectra,objtype.shape[0])-1,"Last spectrum")
	PrimaryImage.header["NSPEC"]=(min(args.nspectra,objtype.shape[0]),"Number of spectra")
	PrimaryImage.header["WAVEMIN"]=(min(armWaves[arm]),"First wavelength [Angstroms]")
	PrimaryImage.header["WAVEMAX"]=(max(armWaves[arm]),"Last wavelength [Angstroms]")
	PrimaryImage.header["WAVESTEP"]=(np.gradient(armWaves[arm])[0],"Wavelength step size [Angstroms]")
	PrimaryImage.header["SPECTER"]=0
	PrimaryImage.header["IN_PSF"]=(".../desimodel/trunk/data/specpsf/psf-"+armName[arm]+".fits","Input spectral PSF")
	PrimaryImage.header["IN_IMG"]=("...im/alpha-3/20150107/pix-"+armName[arm]+"0-"+EXPID+".fits","Input image")

       # HDU1 - Inverse Variance (on Object+Sky)

	NIVARImage.header["NAXIS1"]=(armBins[arm],"Number of wavlengths")
	NIVARImage.header["NAXIS2"]=(args.nspectra,"Number of spectra")
	NIVARImage.header["EXTNAME"]="IVAR"

        # HDU2 - Wavelength

	WaveImage.header["NAXIS1"]=(armBins[arm],"Number of wavelengths")
	WaveImage.header["EXTNAME"]="WAVELENGTH"

	framehdulist=pyfits.HDUList([PrimaryImage,NIVARImage,WaveImage])
	prihdr=framehdulist[0].header
	framehdulist.writeto(framefileName,clobber=True)
	framehdulist.close()

#######------------------skymodel file--------------------------

        skyfileName="skymodel-%s%s-%s.fits"%(armName[arm],spectrograph,EXPID)
	skyImage=pyfits.PrimaryHDU(NSKY[:armBins[arm],arm,:]+SKY_RAND_NOISE[:armBins[arm],arm,:]) # SKY counts+ Random Noise
	skyIvar=pyfits.ImageHDU(data=SKY_IVAR[:armBins[arm],arm,:],name="IVAR")

	#HDU0 - Sky Photons	
	skyImage.header["NAXIS1"]=armBins[arm]
	skyImage.header["NAXIS2"]=args.nspectra
	skyImage.header["CRVAL1"]=(armLimits[arm][0],"Starting wavelength [Angstroms]")
	skyImage.header["CDELT1"]=(np.gradient(armWaves[arm])[0],"Wavelength step size [Angstroms]")
	skyImage.header["EXTNAME"]='SKY_PHOTONS' #What to call this?

	#HDU1 - Inverse Variance(sky)
	skyIvar.header["NAXIS"]=armBins[arm]
	skyIvar.header["EXTNAME"]='IVAR'
	
	skyhdulist=pyfits.HDUList([skyImage,skyIvar])
	prihdr=skyhdulist[0].header
	skyhdulist.writeto(skyfileName,clobber=True)
	skyhdulist.close()

######-------------------------cframe file---------------------------


	cframeFileName="cframe-%s%s-%s.fits"%(armName[arm],spectrograph,EXPID)
	cframeImage=pyfits.PrimaryHDU(CFRAME_OBSERVEDFLUX[:armBins[arm],arm,:])
	fluxIvarImage=pyfits.ImageHDU(data=CFRAME_IVAR[:armBins[arm],arm,:],name="IVAR")
		#maskImage=pyfits.ImageHDU(data=???,name="MASK")

	#HDU0 - Calibrated Flux ( erg/s/cm2/A)
	
        cframeImage.header["NAXIS1"]=armBins[arm]
	cframeImage.header["CRVAL1"]=armLimits[arm][0]
	cframeImage.header["EXTNAME"]=("FLUX","erg/s/cm2/A")
	
        #HDU1 - Inverse Variance in Flux (erg/s/cm2/A)^-2
	fluxIvarImage.header["NAXIS"]=0
	fluxIvarImage.header["EXTNAME"]=('IVAR',"(erg/s/cm2/A)^-2")
	
	cframehduList=pyfits.HDUList([cframeImage,fluxIvarImage])
	prihdr=cframehduList[0].header
	cframehduList.writeto(cframeFileName,clobber=True)
	cframehduList.close()

########--------------------calibration vector file-----------------
	
	calibVectorFile="fluxcalib-%s%s-%s.fits"%(armName[arm],spectrograph,EXPID)
	calibImage=pyfits.PrimaryHDU(CFRAME_OBSERVEDFLUX[:armBins[arm],arm,:]/NOBJ[:armBins[arm],arm,:])

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
	calibhdulist.writeto(calibVectorFile,clobber=True)
	calibhdulist.close()
	
#spectrograph=spectrograph+1	
		

