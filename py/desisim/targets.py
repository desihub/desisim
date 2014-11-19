"""
Utility functions for working with simulated targets
"""

import yaml
import os
import numpy as np
import sys
import fitsio
import random
import desisim.cosmology
import desisim.interpolation
import astropy.units 
import math

def sample_nz(objtype, n):
    """
    Given `objtype` = 'LRG', 'ELG', 'QSO', 'STAR', 'STD'
    return array of `n` redshifts that properly sample n(z)
    from $DESIMODEL/data/targets/nz*.dat
    """
    #- TODO: should this be in desimodel instead?

    #- Stars are at redshift 0 for now.  Could consider a velocity dispersion.
    if objtype in ('STAR', 'STD'):
        return np.zeros(n, dtype=float)
        
    #- Determine which input n(z) file to use    
    targetdir = os.getenv('DESIMODEL')+'/data/targets/'
    objtype = objtype.upper()
    if objtype == 'LRG':
        infile = targetdir+'/nz_lrg.dat'
    elif objtype == 'ELG':
        infile = targetdir+'/nz_elg.dat'
    elif objtype == 'QSO':
        #- TODO: should use full dNdzdg distribution instead
        infile = targetdir+'/nz_qso.dat'
    else:
        raise ValueError("objtype {} not recognized (ELG LRG QSO STD STAR)".format(objtype))
            
    #- Read n(z) distribution
    zlo, zhi, ntarget = np.loadtxt(infile, unpack=True)[0:3]
    
    #- Construct normalized cumulative density function (cdf)
    cdf = np.cumsum(ntarget, dtype=float)
    cdf /= cdf[-1]

    #- Sample that distribution
    x = np.random.uniform(0.0, 1.0, size=n)
    return np.interp(x, cdf, zhi)

    
def sample_targets(nfiber):
    """
    Return tuple of arrays of true_objtype, target_objtype, z
    
    true_objtype   : array of what type the objects actually are
    target_objtype : array of type they were targeted as
    z : true object redshifts
    
    Notes:
    - Actual fiber assignment will result in higher relative fractions of
      LRGs and QSOs in early passes and more ELGs in later passes.
    - This could be expanded to include magnitude and color distributions
    """

    #- Load target densities
    #- TODO: what about nobs_boss (BOSS-like LRGs)?
    fx = open(os.getenv('DESIMODEL')+'/data/targets/targets.dat')
    tgt = yaml.load(fx)
    n = float(tgt['nobs_lrg'] + tgt['nobs_elg'] + \
              tgt['nobs_qso'] + tgt['nobs_lya'] + tgt['ntarget_badqso'])
        
    #- Fraction of sky and standard star targets is guaranteed
    nsky = int(tgt['frac_sky'] * nfiber)
    nstd = int(tgt['frac_std'] * nfiber)
    
    #- Number of science fibers available
    nsci = nfiber - (nsky+nstd)
    
    #- LRGs ELGs QSOs
    nlrg = np.random.poisson(nsci * tgt['nobs_lrg'] / n)
    
    nqso = np.random.poisson(nsci * (tgt['nobs_qso'] + tgt['nobs_lya']) / n)
    nqso_bad = np.random.poisson(nsci * (tgt['ntarget_badqso']) / n)
    
    nelg = nfiber - (nlrg+nqso+nqso_bad+nsky+nstd)
    
    true_objtype  = ['SKY']*nsky + ['STD']*nstd
    true_objtype += ['ELG']*nelg
    true_objtype += ['LRG']*nlrg
    true_objtype += ['QSO']*nqso + ['QSO_BAD']*nqso_bad
    assert(len(true_objtype) == nfiber)
    np.random.shuffle(true_objtype)
    
    target_objtype = list()
    for x in true_objtype:
        if x == 'QSO_BAD':
            target_objtype.append('QSO')
        else:
            target_objtype.append(x)

    target_objtype = np.array(target_objtype)
    true_objtype = np.array(true_objtype)

    #- Fill in z distributions; default 0 for STAR, STD, QSO_BAD
    z = np.zeros(nfiber)
    for objtype in ('ELG', 'LRG', 'QSO'):
        ii = np.where(true_objtype == objtype)[0]
        z[ii] = sample_nz(objtype, len(ii))
    
    return true_objtype, target_objtype, z
    
    
def get_templates(wave, objtype, redshift):
    """
    Return a set of template spectra
    
    Inputs:
    - wave : observer frame wavelength array in Angstroms
    - objtype : array of object types (LRG, ELG, QSO, STAR)
    - redshift : array of redshifts for these objects
    
    Returns 2D array of spectra[len(objtype), len(wave)]
    """

    #- Look for templates in
    #- $DESI_LRG_TEMPLATES, $DESI_ELG_TEMPLATES, etc.
    #- If those envars aren't set, default to most recent version number in
    #- $DESI_SPECTRO_TEMPLATES/{objtype}_templates/{version}/*.fits[.gz]
    
    #- Randomly subsample those templates to get nspec of them    

    
    #- Allow objtype to be a single string instead of an array of strings
    if isinstance(objtype, str):
        objtype = [objtype,] * len(redshift)
    else:
        assert len(redshift) == len(objtype)
    
    #- Store the list of known objtype and associated template filename
    known_filenames_for_objtypes={}
    missing_template_env = list()
    for xobj in set(objtype):
        key = 'DESI_'+xobj+'_TEMPLATES'
        if key in os.environ:
            known_filenames_for_objtypes[key] = os.getenv(key)
        else:
            missing_template_env.append(key)
            
    if len(missing_template_env) > 0:
        raise EnvironmentError("Missing env vars "+str(missing_template_env))
    
    #- allocate spectra
    spectra=np.zeros((len(objtype), len(wave)))
    
    #- open only once each type to minimize IO
    for obj in set(objtype) :
        
        try :
            filename=known_filenames_for_objtypes[obj]
        except KeyError, e:
            print "ERROR in targets.get_templates, unknown objtype '%s'"%obj
            print "known objtypes are",known_filenames_for_objtypes.keys
            raise e
        
        print filename
        file=fitsio.FITS(filename)
        header = file[0].read_header()

        if header["CTYPE1"] != 'WAVE-WAV' or header["CUNIT1"] != 'Angstrom' \
                or  header["BUNIT"] != 'erg/s/cm2/A' \
                or  header["FLUXUNIT"] != 'erg/s/cm2/A' :
            print "ERROR in targets.get_templates, sorry, something has changed in the template files since implementation, need to review this function"
            sys.exit(12) # need a better exception here 
                
        loglam=header["CDELT1"]*np.arange(header["NAXIS1"])+header["CRVAL1"]
        file_restframe_wave=10**loglam
        number_of_spectra_in_file=header["NAXIS2"]
        print loglam.size,number_of_spectra_in_file
        
        #- now loop on requested entries
        for obj_req, z, index in zip(objtype,redshift,np.arange(len(redshift))) :
            
            #- check match objtype of file we have opened 
            if obj_req != obj :
                continue
            
            #- take one random row in file
            entry=int(random.random()*number_of_spectra_in_file)           
            
            #- read only one entry of the file
            restframe_spectrum=file[0][entry:entry+1,:][0]
            # print obj,z,entry,restframe_spectrum.shape
            
            #- use fiducial cosmology to get luminosity distance
            dl=desisim.cosmology.fiducial_cosmology.luminosity_distance(z)
            
            #- ratio of luminosity distance between redshift z and 10pc
            dl_ratio=(10*astropy.units.pc)/desisim.cosmology.fiducial_cosmology.luminosity_distance(z)
            
            #- factor 1/(1+z) because energy density per unit observed wavelength
            scale=dl_ratio**2/(1.+z)

            # just if someone wonders,
            # templates are SEDs at 10pc : phi_0 = ergs/s/cm2/A at d=10pc , z=0
            # luminosity is L (ergs/s/A) = (4*pi*(10pc)^2) * phi_0
            # dd=distance from source , dl=(1+z)*dd=luminosity distance
            # number of photons per unit area (of energy,time interval,wave interval E_e,dt_e,dwave_e at emission)
            # dn_photons = L/(4*pi*(dd(z))^2)/E_e*dt_e*dwave_e
            #            = L/(4*pi*(dd(z))^2)/E_r*dt_r*dwave_r/(1+z)^3
            # phi_obs = observed SED in ergs/s/cm2/A :
            # phi_obs = E_r*dn_photons/(dt_r*dwave_r)
            #         = L/(4*pi*(dd(z))^2)/(1+z)^3
            #         = phi_0*(10pc/dd(z))^2/(1+z)^3
            #         = phi_0*(10pc/(dl(z)/(1+z))^2/(1+z)^3
            #         = phi_0*(10pc/dl(z))^2/(1+z)
            
            
            #- observer frame wave
            file_obsframe_wave=(1.+z)*file_restframe_wave
            
            #- use interpolation routine that conserves flux
            spectra[index]=desisim.interpolation.general_interpolate_flux_density(wave,file_obsframe_wave,scale*restframe_spectrum)
            
    
    return spectra
