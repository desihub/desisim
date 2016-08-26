from desisim.templates import QSO
from desisim.io import empty_metatable
import fitsio
import scipy as sp
from scipy import interpolate

def get_spectra(infile,first=1,last=-1):
 
    '''
    read input lyman-alpha absorption files,
    return normalized spectra including forest absorption.

    Parameters:
    infile: name of the lyman-alpha spectra file
    first: first spectrum to read
    last: last spectrum to read

    returns:
    spectra with forest absorption normalized according to the magnitude
    '''

    h = fitsio.FITS(infile)
    if last<0:
	last = len(h)

    if first<=0:
	print "first must be >= 1"
	return

    heads = [head.read_header() for head in h[first:last]]

    zqso = [head["ZQSO"] for head in heads]
    ra = [head["RA"] for head in heads]
    dec = [head["DEC"] for head in heads]
    mag_g = [head["MAG_G"] for head in heads]

    nqso = len(zqso)

    seed=None
    rand = sp.random.RandomState(seed)
    seed = rand.randint(2**32, size=nqso)

    input_meta = empty_metatable(objtype='QSO', nmodel=nqso)

    qso = QSO()
    qso.normfilter="decam2014-g"
    input_meta["REDSHIFT"]=zqso
    input_meta["RA"]=ra
    input_meta["DEC"]=dec
    input_meta["MAG"]=mag_g
    input_meta["SEED"]=seed

    flux, wave, meta = qso.make_templates(input_meta=input_meta)

    for i,head in enumerate(h[first:last]):
	## read lambda and forest transmission
	la = head["LAMBDA"][:]
	tr = head["FLUX"][:]
	if len(tr)==0:
	    continue

        ## will interpolate the transmission at the spectral wavelengths, 
	## if outside the forest, the transmission is 1
	itr=interpolate.interp1d(la,tr,bounds_error=False,fill_value=1)
	flux[i,:]*=itr(wave)

    h.close()
    return flux,wave,meta


