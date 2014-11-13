"""
Utility functions for working with simulated targets
"""

def sample_nz(objtype, n):
    """
    Given `objtype` = 'LRG', 'ELG', 'QSO', 'STAR', 'STD'
    return array of `n` redshifts that properly sample n(z)
    from $DESIMODEL/data/targets/nz*.dat
    """
    raise NotImplementedError
    
def sample_targets(n):
    """
    Return tuple of arrays of objtype, z
    """
    raise NotImplementedError
    
def get_templates(wave, objtype, redshift):
    """
    Return a set of template spectra
    
    Inputs:
    - wave : observer frame wavelength array in Angstroms
    - objtype : array of object types (LRG, ELG, QSO, STAR)
    - redshift : array of redshifts for these objects
    
    Returns 2D array of spectra[len(objtype), len(wave)]
    """
    
    nspec = len(redshift)

    #- Allow objtype to be a single string instead of an array of strings
    if is_instance(objtype, str):
        objtype = [objtype,] * nspec
    else:
        assert len(redshift) == len(objtype)
        
    #- Look for templates in
    #- $DESI_LRG_TEMPLATES, $DESI_ELG_TEMPLATES, etc.
    #- If those envars aren't set, default to most recent version number in
    #- $DESI_SPECTRO_TEMPLATES/{objtype}_templates/{version}/*.fits[.gz]
    
    #- Randomly subsample those templates to get nspec of them
        
    raise NotImplementedError
    