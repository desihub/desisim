"""
desisim.survey_release
=============

Functions and methods for mimicking a DESI release footprint and object density with adequate exposure times.

"""
import os
import desisim
import numpy as np
import healpy as hp
import fitsio
from astropy.table import Table, vstack
from desiutil.log import get_logger
from scipy.stats import rv_discrete
from desitarget.targetmask import desi_mask
from desimodel.io import load_tiles
from desimodel.footprint import tiles2pix, is_point_in_desi, radec2pix
from scipy.spatial import cKDTree


log = get_logger()

NSIDE_DENSITY, NSIDE_POS = 256, 128                 # healpix nside of the footprint weight and of the NTILE / exposure-time histograms
POOL_MARGIN = 1.1                               # extra oversizing of the non QSO candidate pool, against its Poisson noise
SECONDS_PER_PASS = 1000.                        # one nominal pass: quickquasars simulates perfect dark conditions
NTILE_MAX = 9                                   # NTILE histogram columns 0..9 (values above are clipped)
EXPTIME_BIN_WIDTH = 500.                            # exposure-time histogram bin width [s]
N_EXPTIME_BINS = 24                                 # 0 - 12000 s
# exposure time [s] = factor * TSNR2 column; every TSNR2 type has its own factor (in order of preference)
TSNR_EXPTIME_FACTORS = {'TSNR2_LRG': 12.15, 'TSNR2_LYA': 11.8, 'TSNR2_QSO': 33.61}
TSNR_COLUMN, EXPTIME_FACTOR = next(iter(TSNR_EXPTIME_FACTORS.items()))


def round_to_passes(exptime, seconds_per_pass=SECONDS_PER_PASS):
    """Round an exposure time [s] to the nearest whole multiple of ``seconds_per_pass`` (minimum one multiple)."""
    exptime = np.asarray(exptime, dtype=float)
    if not seconds_per_pass:
        return exptime
    return seconds_per_pass * np.maximum(1., np.floor(exptime / seconds_per_pass + 0.5))



class LyaCoverage(object):
    """The Lya coverage of a data release: what ``gen_qso_catalog`` needs to give the mock Lya quasar targets the footprint, completeness
    and exposure times of the data. Built once per release by ``build_lya_coverage`` from the LSS randoms and stored as
    ``lya_<release>_coverage.fits``.

    Dense healpix arrays (footprint weight, histogram of the number of available tiles NTILE, exposure histograms) plus a few numbers fitted
    once to the data catalog: q and N_DATA (QSO targets), q_nonqso and n_data_nonqso (non-QSO targets: quasars selected as ELG or other
    classes), p_reobs (header only) and the pass time. No sky position of the data enters it.

    Mock selection: keep probability = kappa * w_pix * P_obs(pix; q), with w_pix the footprint weight from the number of randoms per pixel
    (``nside_density``) and P_obs = sum_n h_n [1 - (1 - q)^n] / sum_n h_n from the NTILE histogram h_n of the pixel (``nside_pos``); kappa is
    solved per target class so that the expected number of kept mock targets equals the number of data targets of that class.
    Exposures are drawn from the exposure histogram of the pixel and rounded to whole nominal passes (``seconds_per_pass`` = 1000 s, minimum one pass:
    quickquasars simulates perfect dark conditions, so one nominal exposure is 1000 s and the data effective time 12.15 * TSNR2_LRG is its
    nominal-dark equivalent).
    """

    def __init__(self, acc, nside_density=NSIDE_DENSITY, nside_pos=NSIDE_POS, q=0.5, n_data=None, meta=None,
                 interior_threshold=0.75, min_footprint=0.1, seconds_per_pass=0., q_nonqso=None, n_data_nonqso=None):
        self.nside_density, self.nside_pos = nside_density, nside_pos
        self.seconds_per_pass, self.q_nonqso, self.n_data_nonqso = seconds_per_pass, q_nonqso, n_data_nonqso
        self.acc = acc; self.q, self.n_data = q, n_data
        self.meta = dict(meta or {})
        self.interior_threshold, self.min_footprint = interior_threshold, min_footprint
        nr = acc['nr_density'].astype(float)
        occupied = nr[nr > 0]
        self.n_full = np.percentile(occupied, 95)          # randoms in a fully covered pixel
        w = np.clip(nr / self.n_full, 0, 1)
        w[nr >= interior_threshold * self.n_full] = 1.0    # no noise in the interior
        w[nr < min_footprint * self.n_full] = 0.0
        self.weight = w
        self._pobs_q = None

    def pobs(self, q=None):
        """Probability of being observed at least once, per nside_pos pixel, from the NTILE histogram."""
        q = self.q if q is None else q
        if self._pobs_q != q:
            h = self.acc['h_ntile'].astype(float); n = np.arange(h.shape[1])
            tot = h.sum(1)
            self._pobs = np.divide(h @ (1 - (1 - q) ** n), tot, out=np.zeros(len(tot)), where=tot > 0)
            self._pobs_q = q
        return self._pobs

    def nonqso_pool_factor(self):
        """Oversizing of the pool of non QSO candidates (relative to the data density) that the selection of non QSO targets needs.

        Candidates have the data density times this factor, so kappa = 1 / (factor * <P_obs>) with <P_obs> the footprint-weighted mean of P_obs.
        No pixel saturates (kappa * w * P_obs <= 1) when factor >= max(P_obs) / <P_obs>; the value returned is that ratio (at least 1) times POOL_MARGIN,
        and 1 without non QSO targets.
        """
        if self.q_nonqso is None:
            return 1.0
        ratio = (self.nside_density // self.nside_pos) ** 2
        area = self.weight.reshape(-1, ratio).sum(1)            # footprint weight inside each nside_pos pixel
        pobs = self.pobs(self.q_nonqso)
        return POOL_MARGIN * max(1.0, float(pobs[area >= 0.5 * ratio].max() / (np.sum(area * pobs) / np.sum(area))))

    def pixels(self, ra, dec):
        p_d = hp.ang2pix(self.nside_density, np.radians(90 - np.asarray(dec)), np.radians(ra), nest=True)
        return p_d, p_d // (self.nside_density // self.nside_pos) ** 2

    def keep_probability(self, ra, dec, kappa=1.0, q=None):
        p_d, p_p = self.pixels(ra, dec)
        return np.minimum(1.0, kappa * self.weight[p_d] * self.pobs(q)[p_p])

    def solve_kappa(self, ra, dec, n_target, q=None):
        """kappa such that the expected number of kept targets equals n_target (completeness slope q, default the QSO-target one)."""
        p_d, p_p = self.pixels(ra, dec); x = self.weight[p_d] * self.pobs(q)[p_p]
        if np.sum(np.minimum(1.0, 1e6 * x)) < n_target:
            log.warning(f"Not enough mock targets ({np.sum(x > 0)}) to reach the requested {n_target}; keeping all of them.")
            return 1e6
        lo, hi = 0.0, 1.0
        while np.sum(np.minimum(1.0, hi * x)) < n_target:
            hi *= 2
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if np.sum(np.minimum(1.0, mid * x)) < n_target: lo = mid
            else: hi = mid
        return 0.5 * (lo + hi)

    def exptime_cdf(self):
        if not hasattr(self, '_cdf'):
            h = self.acc['h_exp'].astype(float); tot = h.sum(1)
            self._cdf = np.cumsum(h, axis=1) / np.where(tot > 0, tot, 1)[:, None]
            self._filled = tot > 0
            if not np.any(self._filled):
                raise RuntimeError("Lya coverage without any exposure histogram")
            # pixels with randoms but no observed target (edges) use the nearest filled pixel
            # nearest filled pixel for lookups that land on an empty one
            centers = np.array(hp.pix2vec(self.nside_pos, np.arange(len(tot)), nest=True)).T
            tree = cKDTree(centers[self._filled]); idx_filled = np.flatnonzero(self._filled)
            self._source = np.arange(len(tot))
            empty = np.flatnonzero(~self._filled)
            if len(empty): self._source[empty] = idx_filled[tree.query(centers[empty])[1]]
        return self._cdf

    def draw_exptimes(self, ra, dec, rng, chunk=300_000, seconds_per_pass=None):
        """Draw one exposure [s] per position from its pixel histogram (uniform jitter inside the bin),
        rounded to whole multiples of ``seconds_per_pass`` (default: the seconds per pass of the map; 0 = continuous)."""
        seconds_per_pass = self.seconds_per_pass if seconds_per_pass is None else seconds_per_pass
        cdf = self.exptime_cdf()
        p_p = self.pixels(ra, dec)[1]; p_p = self._source[p_p]
        out = np.zeros(len(p_p))
        for a in range(0, len(p_p), chunk):
            sl = slice(a, a + chunk)
            u = rng.uniform(size=len(p_p[sl]))
            b = np.minimum((cdf[p_p[sl]] < u[:, None]).sum(1), N_EXPTIME_BINS - 1)
            out[sl] = (b + rng.uniform(size=len(b))) * EXPTIME_BIN_WIDTH
        return round_to_passes(out, seconds_per_pass)

    # ---- IO
    def write(self, filename):
        a = self.acc
        hdr = dict(NSIDED=self.nside_density, NSIDEP=self.nside_pos, Q=self.q, NDATA=self.n_data if self.n_data else -1,
                   NFULL=self.n_full, INTERIOR=self.interior_threshold, MINFOOT=self.min_footprint, EXPBIN=EXPTIME_BIN_WIDTH,
                   SECPPASS=self.seconds_per_pass, QNQ=self.q_nonqso if self.q_nonqso is not None else -1.,
                   NDATANQ=self.n_data_nonqso if self.n_data_nonqso else -1, **self.meta)
        import fitsio
        sel_d = np.flatnonzero(a['nr_density'] > 0); sel_p = np.flatnonzero(a['nr_pos'] > 0)
        with fitsio.FITS(filename, 'rw', clobber=True) as f:
            f.write(dict(PIXEL=sel_d.astype('i8'), NR=a['nr_density'][sel_d].astype('i4')), extname='DENSITY', header=hdr)
            f.write(dict(PIXEL=sel_p.astype('i8'), NR=a['nr_pos'][sel_p].astype('i4'),
                         H_NTILE=a['h_ntile'][sel_p].astype('i4'), H_EXP=a['h_exp'][sel_p].astype('i4')), extname='POSITION', header=hdr)

    @classmethod
    def read(cls, filename):
        import fitsio
        with fitsio.FITS(filename) as f:
            hdr = f['DENSITY'].read_header(); dd = f['DENSITY'].read(); pp = f['POSITION'].read()
        nd, npos = int(hdr['NSIDED']), int(hdr['NSIDEP'])
        acc = dict(nr_density=np.zeros(12 * nd ** 2, np.int64), nr_pos=np.zeros(12 * npos ** 2, np.int64),
                   h_ntile=np.zeros((12 * npos ** 2, NTILE_MAX + 1), np.int64), h_exp=np.zeros((12 * npos ** 2, N_EXPTIME_BINS), np.int64))
        acc['nr_density'][dd['PIXEL']] = dd['NR']; acc['nr_pos'][pp['PIXEL']] = pp['NR']
        acc['h_ntile'][pp['PIXEL']] = pp['H_NTILE']; acc['h_exp'][pp['PIXEL']] = pp['H_EXP']
        n_data = int(hdr['NDATA']); n_data = None if n_data < 0 else n_data
        get = lambda key, default: hdr[key] if key in hdr else default
        qnq = float(get('QNQ', -1.)); nnq = int(get('NDATANQ', -1))
        meta = {key: hdr[key] for key in ('RELEASE', 'ZMIN', 'ZMAX') if key in hdr}        # release and redshift range of the counted data targets (absent in older files)
        return cls(acc, nd, npos, q=float(hdr['Q']), n_data=n_data, meta=meta,
                   interior_threshold=float(hdr['INTERIOR']), min_footprint=float(hdr['MINFOOT']),
                   seconds_per_pass=float(get('SECPPASS', 0.)), q_nonqso=None if qnq < 0 else qnq, n_data_nonqso=None if nnq < 0 else nnq)




class SurveyRelease(object):
    """ Class object to preprocess a master catalog to mimic a DESI release footprint and object density with adequate exposure times.

    Args:
        mastercatalog (str): path to the master catalog.
        data_file (str): path to the data catalog.
        include_nonqso_targets (bool): if True, will keep non QSO targets on data catalog and generate them in mock catalog.
        seed (int): seed for random number generation.
        invert (bool): if True, will invert the random number generation for redshift and magnitude distributions.
        nonqso_pool_factor (float): multiplies the number of non QSO candidates drawn from the master (default 1: as many as the data has per area).
            The Lya coverage selection keeps only a fraction of them (completeness proportional to the number of available tiles), so it needs a larger pool
            (gen_qso_catalog uses LyaCoverage.nonqso_pool_factor()).
    """
    def __init__(self,mastercatalog,data_file=None,include_nonqso_targets=False,seed=None,invert=False,nonqso_pool_factor=1.0):
        self.mockcatalog=Table.read(mastercatalog,hdu=1) # Assumes master catalog is in HDU1
        log.info(f"There are {len(self.mockcatalog)} targets in {mastercatalog} master catalog.")
        if 'Z' not in self.mockcatalog.colnames:
            if 'Z_QSO_RSD' in self.mockcatalog.colnames:
                self.mockcatalog.rename_column('Z_QSO_RSD', 'Z')
            else:
                raise Exception("Mock catalog Z column not found")
        self.invert=invert
        self.include_nonqso_targets = include_nonqso_targets
        self.nonqso_pool_factor = nonqso_pool_factor
        np.random.seed(seed)

        self.data = None
        self.footprint_method = None     # 'tiles' or 'coverage' once apply_data_geometry / apply_lya_coverage has been applied; assign_exposures follows it
        if data_file is not None:
            self.data = self.prepare_data_catalog(data_file,zmin=min(self.mockcatalog['Z']),zmax=max(self.mockcatalog['Z']),
                                                  include_nonqso_targets=self.include_nonqso_targets)
        elif self.include_nonqso_targets:
            raise ValueError("The include_nonqso_targets option was set to True. This requires an input data catalog to extract distributions.")
            
    def _uniform(self,size):
        """ Uniform random numbers in [0,1), flipped when the selection is inverted. """
        rand = np.random.uniform(size=size)
        return 1-rand if self.invert else rand

    def _data_radec(self):
        """ Positions (ra, dec) of the data catalog: TARGET_RA, TARGET_DEC if present, otherwise RA, DEC. """
        colnames = self.data.colnames
        if 'TARGET_RA' in colnames and 'TARGET_DEC' in colnames:
            return self.data['TARGET_RA'], self.data['TARGET_DEC']
        if 'RA' in colnames and 'DEC' in colnames:
            return self.data['RA'], self.data['DEC']
        raise ValueError("No RA,DEC or TARGET_RA,TARGET_DEC columns in data catalog")

    @staticmethod
    def prepare_data_catalog(cat_filename,zmin=None,zmax=None,include_nonqso_targets=False):
        """Prepare the data catalog for the quickquasars pipeline.
        Args:
            cat_filename (str): path to the data catalog.
            zmin (float): minimum redshift of the distribution
            zmax (float): maximum redshift of the distribution
            include_nonqso_targets (bool): if True, will keep non QSO targets on data catalog and generate them in mock catalog.
        returns:
            cat (astropy.table.Table): data catalog preprocessed in the redshift range of the mock master catalog.
        """
        log.info(f"Reading data catalog: {cat_filename}")
        cat = Table.read(cat_filename)
        log.info(f"Found {len(cat)} targets in data catalog.")
        w_z =(cat['Z']>=zmin)&(cat['Z']<=zmax)
        log.info(f"Keeping {np.sum(w_z)} targets in redshift range ({zmin},{zmax}) in data catalog.")
        cat=cat[w_z]
        
        qso_targets_mask = get_qsotarget_mask(cat)
        log.info(f"Keeping {np.sum(qso_targets_mask)} ({np.sum(qso_targets_mask)/np.sum(w_z):.2%}) QSO targets in data catalog.")
        if not include_nonqso_targets:
            log.info(f"Excluding {np.sum(~qso_targets_mask)} ({np.sum(~qso_targets_mask)/np.sum(w_z):.2%}) non QSO targets from data catalog.")
            cat=cat[qso_targets_mask]
        else:
            log.info(f"Keeping {np.sum(~qso_targets_mask)} ({np.sum(~qso_targets_mask)/np.sum(w_z):.2%}) non QSO targets in data catalog.")
        return cat
    
    def apply_redshift_dist(self,zmin=0,zmax=10,dz=0.1,distribution='SV'):
        """Apply a redshift distribution to the catalog.

        Args:
            zmin (float): minimum redshift of the distribution
            zmax (float): maximum redshift of the distribution
            dz (float): redshift bin width
            distribution (str): 'SV' or 'target_selection' for the moment.
        """
        if distribution=='SV':
            filename = os.path.join(os.path.dirname(desisim.__file__),'data/dn_dzdM_EDR.fits')
            with fitsio.FITS(filename) as fts:
                dn_dzdm=fts[3].read()
                zcenters=fts[1].read()

            dndz=np.sum(dn_dzdm,axis=1)
            dz = zcenters[1] - zcenters[0]
            zmin = max(zcenters[0]-0.5*dz,zmin)
            zmax = min(zcenters[-1]+0.5*dz,zmax)
            w_z = (zcenters-0.5*dz > zmin) & (zcenters+0.5*dz <= zmax)
            dndz=dndz[w_z]
            zcenters=zcenters[w_z]
            zbins = np.linspace(zmin,zmax,len(zcenters)+1)
        elif distribution=='target_selection':
            dist = Table.read(os.path.join(os.path.dirname(desisim.__file__),'data/redshift_dist_chaussidon2022.ecsv')) 
            dz_dist=dist['z'][1]-dist['z'][0]
            factor=0.1/dz_dist # Account for redshift bin width difference.
            
            zbins = np.linspace(zmin,zmax,int((zmax-zmin)/dz)+1)
            zcenters=0.5*(zbins[1:]+zbins[:-1])
            dndz=factor*np.interp(zcenters,dist['z'],dist['dndz_23'],left=0,right=0)
        elif distribution=='from_data':
            raise NotImplementedError(f"Option {distribution} is not implemented yet")
        else:
            raise ValueError(f"Distribution option {distribution} not in available options: SV, target_selection, from_data")
        
        selection = self.sample_redshift(dndz,zbins=zbins)
        selected_ids = list(self.mockcatalog['MOCKID'][selection])
        self.mockcatalog['IS_QSO_TARGET'] = np.isin(self.mockcatalog['MOCKID'],selected_ids)
        log.info(f"Selecting {sum(selection)} mock QSO targets following the distribution: {distribution}")
        if self.include_nonqso_targets:
            avail_ids = self.mockcatalog['MOCKID'][~selection]
            dndz_data_nonqso = self.nonqso_pool_factor*self.get_nonqso_dndz(expected_dist=dist,zbins=zbins) 
            selected_nonqso = self.sample_redshift(dndz_data_nonqso,zbins=zbins,mask=~selection)
            selected_nonqso_ids = avail_ids[selected_nonqso]
            log.info(f"Selecting {sum(selected_nonqso)} mock non QSO targets following the distribution of data.")
            self.are_nonqso_targets = np.isin(self.mockcatalog['MOCKID'],selected_nonqso_ids)
            selection |= self.are_nonqso_targets  
        log.info(f"Selected {sum(selection)} mock targets out of {len(self.mockcatalog)}")
        self.mockcatalog=self.mockcatalog[selection]
                                 
    def get_nonqso_dndz(self,expected_dist, zbins=np.linspace(0,10,100+1)):
        """Get the redshift distribution of non QSO targets in the data catalog.

        Args:
            expected_dist (Table): Table with expected redshift distribution.
            zbins (array): redshift bins.

        Returns:    
            dndz_data_nonqso (array): redshift distribution of non QSO targets in the data catalog.
        """
        log.info(f"Sampling mock targets from data non QSO targets. Assuming data QSO targets already follow expected distribution.")
        targetmask = get_qsotarget_mask(self.data)
        nonqso_targets = self.data[~targetmask]
        dndz_data_nonqso = np.histogram(nonqso_targets['Z'],bins=zbins)[0]
        # Figure out the area of the catalog by checking number of Lya. 
        # This assumes that data lya qso targets already follow the expected distribution.
        qso_targets = self.data[targetmask]
        lyabins = np.linspace(2.1,3.8,18)
        dndz_data_qso = np.histogram(qso_targets['Z'],bins=lyabins)[0]
        dz_dist=expected_dist['z'][1]-expected_dist['z'][0]
        factor=0.1/dz_dist # Account for redshift bin width difference.
        zcenters=0.5*(lyabins[1:]+lyabins[:-1])
        expected_dndz=factor*np.interp(zcenters,expected_dist['z'],expected_dist['dndz_23'],left=0,right=0)
        area = np.sum(dndz_data_qso)/np.sum(expected_dndz)
        return dndz_data_nonqso/area
            
    def sample_redshift(self,target_dndz, zbins = np.linspace(0,10,100+1), mask=None):
        """Randomly sample mockcatalog to follow target redshift distribution. 
        
        Args:
            target_dndz (array): target redshift distribution.
            zbins (array): redshift bins.
            mask (array): mask to apply to the catalog before sampling.
            
        Returns:
            selection (array): boolean array with True where the targets were selected."""
        mock_area=get_catalog_area(self.mockcatalog,nside=16)
        mockz = self.mockcatalog['Z']
        if mask is not None:
            mockz = mockz[mask]
        dndz_mock=np.histogram(mockz,bins=zbins,
                                   weights=np.repeat(1/mock_area,len(mockz)))[0]
        fractions=np.divide(target_dndz,dndz_mock,where=dndz_mock>0,out=np.zeros_like(dndz_mock))
        fractions[fractions>1]=1

        bin_index = np.digitize(mockz, zbins) - 1
        selection = self._uniform(mockz.size) < fractions[bin_index]
        return selection
        
    def apply_data_geometry(self, release='Y5',tilefile=None):
        """ Apply the geometry of a given data release to the mock catalog.
        
        Args:
            release (str): data release name. release='Y5' will simulate DESI-Y5 footprint without NPASS downsampling.
            tilefile (str): path to a tile file. If None, will use the tiles from the release.
        """
        mock=self.mockcatalog
        if tilefile is None:
            log.info(f"Getting Lya tiles for {release.upper()} release.")
            tiles=get_lya_tiles(release)
        else:
            log.info(f"Reading tiles from file: {tilefile}")
            tiles=Table.read(tilefile)
        mask_footprint=is_point_in_desi(tiles,mock['RA'],mock['DEC'])
        if sum(mask_footprint)==0:
            raise ValueError("There are no mock targets available in the tiles region")
            
        mock=mock[mask_footprint]
        log.info(f"Keeping {sum(mask_footprint)} mock targets in tiles footprint.")

        # If 'Y5' is selected there is no need to downsample by NPASS the mock catalog.
        if release!='Y5':
            log.info(f"Downsampling by NPASSES fraction in tiles")
            self.npass_pixmap=npass_pixmap_from_tiles(tiles,nside=1024)
            mock_pixels = radec2pix(ra=mock['RA'], dec=mock['DEC'], nside=1024)
            if self.data is None:
                raise ValueError("No data catalog was provided.")
            data_ra, data_dec = self._data_radec()
            data_pixels = radec2pix(ra=data_ra, dec=data_dec, nside=1024)

            data_passes = self.npass_pixmap[data_pixels]['NPASS']
            mock_passes = self.npass_pixmap[mock_pixels]['NPASS']
            data_pass_counts = np.bincount(data_passes,minlength=8) # Minlength = 7 passes + 1 (for bining)
            mock_pass_counts = np.bincount(mock_passes,minlength=8)
            mock['NPASS'] = mock_passes
            downsampling=np.divide(data_pass_counts,mock_pass_counts,out=np.zeros(8),where=mock_pass_counts>0)
            rand = self._uniform(len(mock))
            selection = rand<downsampling[mock_passes]
            log.info(f"Keeping {len(mock[selection])} mock targets out of {len(mock)}.")
            mock = mock[selection]
        self.mockcatalog=mock
        self.footprint_method = 'tiles'

    def apply_lya_coverage(self, lya_coverage):
        """Footprint and downsampling from the Lya coverage of the release (see LyaCoverage and the build_lya_coverage script).

        Replaces apply_data_geometry: mock targets are kept with probability kappa*w*P_obs, with kappa solved per target class so that the
        expected number of kept targets inside the redshift range of the map (ZMIN, ZMAX in its header; all targets if the map does not record
        it) equals the number of data targets of that class stored in the map. QSO targets and non QSO targets (quasars selected as other
        target classes) have their own completeness slope and count.

        Args:
            lya_coverage (str or LyaCoverage): path to lya_<release>_coverage.fits or an already loaded LyaCoverage.
        """
        lmap = LyaCoverage.read(lya_coverage) if isinstance(lya_coverage,str) else lya_coverage
        if lmap.n_data is None:
            raise ValueError("The Lya coverage has no N_DATA scalar; build it with build_lya_coverage and the data catalog.")
        self.lya_coverage = lmap
        mock = self.mockcatalog
        rand = self._uniform(len(mock))
        selection = np.zeros(len(mock),dtype=bool)
        is_qso = np.asarray(mock['IS_QSO_TARGET']) if 'IS_QSO_TARGET' in mock.colnames else np.ones(len(mock),dtype=bool)
        classes = [('QSO',is_qso,lmap.q,lmap.n_data)]
        if np.any(~is_qso):
            if lmap.q_nonqso is None or lmap.n_data_nonqso is None:
                raise ValueError("The mock catalog has non QSO targets but the Lya coverage has no q_nonqso / N_DATA_nonqso; rebuild the map with a recent build_lya_coverage.")
            classes.append(('non QSO',~is_qso,lmap.q_nonqso,lmap.n_data_nonqso))
        # The data counted in the map (N_DATA) are those inside its redshift range [ZMIN, ZMAX] (header of the map; older maps do not record it).
        # Mock redshifts are moved a little outside that range by the FoG, so kappa is solved on the mock targets inside it: the number of kept
        # targets inside the range then equals the data's. The targets outside it get the same kappa.
        zmin, zmax = lmap.meta.get('ZMIN'), lmap.meta.get('ZMAX')
        in_range = np.ones(len(mock),dtype=bool) if zmin is None or zmax is None else (mock['Z']>=zmin)&(mock['Z']<=zmax)
        for name,sel,q,n_data in classes:
            fit = sel&in_range
            kappa = lmap.solve_kappa(mock['RA'][fit],mock['DEC'][fit],n_data,q=q)
            keep = lmap.keep_probability(mock['RA'][sel],mock['DEC'][sel],kappa=kappa,q=q)
            selection[sel] = rand[sel]<keep
            solved_on = f"solved on {np.sum(fit)} of the {np.sum(sel)} mock targets inside the redshift range {zmin}-{zmax}" if zmin is not None and zmax is not None \
                else "solved on all of them: the map does not record a redshift range"
            log.info(f"Lya coverage selection of {name} targets (q={q:.3f}, kappa={kappa:.4g}, {solved_on}): keeping {np.sum(selection[sel])} of {np.sum(sel)} "
                     f"mock targets (data: {n_data}).")
        self.mockcatalog = mock[selection]
        self.footprint_method = 'coverage'

    def assign_rband_magnitude(self,from_data=False):
        """Assign r-band magnitude and include FLUX_R to the catalog according to the distribution.
        
        Args:
            from_data (bool): if True, will use the data catalog to generate the distribution. Otherwise, will use the default distribution as expected by SV.
        """
        if self.include_nonqso_targets and not from_data:
            log.warning("Setting from_data = True as the include_nonqso_targets requires it.") 
            from_data = True
        mock_mask = self.mockcatalog['IS_QSO_TARGET']
        self.mockcatalog['FLUX_R'] = 0. # Initiate FLUX_R values
        if not from_data:
            filename=os.path.join(os.path.dirname(desisim.__file__),'data/dn_dzdM_EDR.fits')
            with fitsio.FITS(filename) as fts:
                zcenters=fts['Z_CENTERS'].read()
                rmagcenters=fts['RMAG_CENTERS'].read()
                dn_dzdm=fts['dn_dzdm'].read()
            log.info(f"Generating random magnitudes according to distribution: {filename}")
        else:
            if self.data is None:
                raise ValueError("No data catalog was provided") 
            data_mask = get_qsotarget_mask(self.data)
            log.info(f"Computing dN/dzdm from data QSO targets")
            dn_dzdm, zbins, rmagbins = get_catalog_dndzdm(self.data,mask=data_mask)
            zcenters=0.5*(zbins[1:]+zbins[:-1])
            rmagcenters=0.5*(rmagbins[1:]+rmagbins[:-1])
            
            if self.include_nonqso_targets:
                log.info(f"Computing dN/dzdm from data non QSO targets")
                dn_dzdm_nonqso, zbins, rmagbins = get_catalog_dndzdm(self.data, mask=~data_mask)
                log.info(f"Assigning r-band magnitude to {sum(~mock_mask)} non QSO targets.")
                self.mockcatalog['FLUX_R'][~mock_mask] = generate_random_fluxes(dn_dzdm_nonqso, self.mockcatalog['Z'][~mock_mask],
                                                                              zcenters, rmagcenters, self.invert) 
                
        log.info(f"Assigning r-band magnitude to {sum(mock_mask)} QSO targets.")
        self.mockcatalog['FLUX_R'][mock_mask] = generate_random_fluxes(dn_dzdm, self.mockcatalog['Z'][mock_mask], zcenters, rmagcenters, self.invert)
        assert np.all(self.mockcatalog['FLUX_R']!=0)
            
    def get_data_exptime(self):
        """ Effective exposure time [s] of the data catalog: factor * TSNR2 column, the first of TSNR_EXPTIME_FACTORS found. """
        for k, (column, factor) in enumerate(TSNR_EXPTIME_FACTORS.items()):
            if column in self.data.colnames:
                if k > 0:
                    log.warning(f'Effective exposure time {TSNR_COLUMN} column not found in observed data catalog.')
                    log.warning('Will compute effective exposure time from alternative templates.')
                log.info(f'Getting effective exposure time in data catalog by {factor}*{column}.')
                return factor*self.data[column]
        raise ValueError(f"Can't compute effective exposure time. Data catalog columns should include {', '.join(TSNR_EXPTIME_FACTORS)}.")

    def assign_exposures(self,exptime=None,method=None,lya_coverage=None):
        """ Assign exposures to the catalog according to the distribution in the data release.

        Args:
            exptime (float, optional): if not None, will assign the given exposure time in seconds to all objects in the mock catalog.
            method (str, optional): 'tiles' draws from the data exposure distribution of each NPASS class (needs apply_data_geometry).
                'coverage' draws from the exposure histogram of the pixel in the Lya coverage: whole nominal passes (multiples of 1000 s) for the Lya QSO
                targets (z>2.1), 1000 s for the other targets. By default the method of the footprint step already applied (apply_lya_coverage: 'coverage';
                apply_data_geometry: 'tiles'), or 'tiles' if none was applied. A method that differs from that of the footprint step raises ValueError.
            lya_coverage (str or LyaCoverage): Lya coverage for method='coverage'. Defaults to the one used in apply_lya_coverage. The exposures are rounded to the
                pass time stored in the map (whole nominal passes of 1000 s, set when the map is built).
        """
        if method not in (None,'tiles','coverage'):
            raise ValueError(f"method must be 'tiles' or 'coverage', got {method}")
        if exptime is None and self.footprint_method is not None and method not in (None,self.footprint_method):
            raise ValueError(f"method='{method}' does not match the footprint step already applied ('{self.footprint_method}').")
        if method is None:
            method = self.footprint_method or 'tiles'
        if exptime is not None:
            log.info(f"Assigning uniform exposures {exptime} seconds")
            self.mockcatalog['EXPTIME']=exptime
        elif method=='coverage':
            self._assign_exposures_from_coverage(lya_coverage)
        else:
            self._assign_exposures_from_tiles()

    def _assign_exposures_from_coverage(self,lya_coverage=None):
        """ Whole nominal passes drawn from the exposure-time histogram of the Lya coverage for the Lya QSO targets, 1000 s for the rest. """
        lmap = lya_coverage if lya_coverage is not None else getattr(self,'lya_coverage',None)
        if lmap is None:
            raise ValueError("method='coverage' requires a lya_coverage (or a previous apply_lya_coverage call).")
        if isinstance(lmap,str):
            lmap = LyaCoverage.read(lmap)
        log.info("Assigning exposures from the Lya coverage.")
        mock = self.mockcatalog
        is_lya = (mock['Z']>2.1)&mock['IS_QSO_TARGET']
        exptime_mock = np.full(len(mock),1000.)
        exptime_mock[is_lya] = lmap.draw_exptimes(mock['RA'][is_lya],mock['DEC'][is_lya],np.random.default_rng(np.random.randint(2**31)))
        self.mockcatalog['EXPTIME'] = exptime_mock

    def _assign_exposures_from_tiles(self):
        """ Exposure times of the Lya QSO targets drawn from the data distribution of each NPASS class, 1000 s for the rest. """
        if self.data is None:
            raise ValueError("No data catalog nor exptime was provided. Please provide one of them")
        log.info("Assigning exposures")
        mock=self.mockcatalog
        data_ra, data_dec = self._data_radec()
        data_pixels = radec2pix(ra=data_ra, dec=data_dec, nside=1024)
        data_mask = (self.data['Z']>2.1)&get_qsotarget_mask(self.data)
        exptime_data = self.get_data_exptime()[data_mask]
        data_passes = self.npass_pixmap[data_pixels]['NPASS'][data_mask]
        mock_mask = (mock['Z']>2.1)&mock['IS_QSO_TARGET']
        exptime_mock = np.zeros(len(mock))
        exptime_mock[~mock_mask]=1000
        for tile_pass in range(1,8):
            pdf=np.histogram(exptime_data[data_passes == tile_pass]/1000,bins=np.arange(.5,9.5,1),density=True)[0]
            random_variable = rv_discrete(values=(np.arange(1,len(pdf)+1),pdf))
            is_pass = mock_mask&(mock['NPASS'] == tile_pass)
            exptime_mock[is_pass]=1000*random_variable.rvs(size=np.sum(is_pass))
        self.mockcatalog['EXPTIME']=exptime_mock

def get_qsotarget_mask(cat):
    """Get boolean mask for QSO and non QSO targets in the data catalog.

    Args:
        cat (astropy.table.Table): data catalog.

    Returns:
        qso_targets_mask (array): boolean mask for QSO targets.
    """
    mask = desi_mask.mask('QSO|QSO_HIZ|QSO_NORTH|QSO_SOUTH')
    qso_targets_mask = ((cat["DESI_TARGET"]&mask)>0)
    return qso_targets_mask
            
def get_catalog_area(catalog, nside=256):
    """Return the area of the catalog in square degrees.

    Args:
        nside (int): HEALPix nside parameter

    Returns:
        area (float): area of the catalog in square degrees.
    """
    if 'DEC' in catalog.colnames and 'RA' in catalog.colnames: 
        pixels = radec2pix(ra=catalog['RA'], dec=catalog['DEC'], nside=nside)
    elif 'TARGET_DEC' in catalog.colnames and 'TARGET_RA' in catalog.colnames:
        pixels = radec2pix(ra=catalog['TARGET_RA'], dec=catalog['TARGET_DEC'], nside=nside)
    else:
        raise ValueError("No RA,DEC or TARGET_RA,TARGET_DEC columns in catalog")
    pixarea = hp.pixelfunc.nside2pixarea(nside, degrees=True)
    npix = len(np.unique(pixels))
    return npix*pixarea

def get_lya_tiles(release='Y5'):
    """Return the tiles that have been observed in a given release.

    Args:
        release (str,optional): release name. If release=='Y5', will return all DESI DARK tiles.

    Returns:
        tiles (astropy.table.Table): Table with the tiles observed in the specified release. 
    """
    all_tiles = load_tiles()
    dark_tiles = all_tiles['PROGRAM']=='DARK'
    tiles = all_tiles[dark_tiles]
    if release!='Y5':
        if release.upper() not in ['FUJI','FUGU']:
            surveys = ['main']
            redux_tiles_filename = os.path.join(os.environ['DESI_SPECTRO_REDUX'],f'{release}/tiles-{release}.fits')
            if not os.path.exists(redux_tiles_filename):
                raise ValueError(f"Tiles file for release {release} not found at {redux_tiles_filename}")
            redux_tiles=Table.read(redux_tiles_filename)
        else:
            fuji_tiles_filename= os.path.join(os.environ['DESI_SPECTRO_REDUX'],f'fuji/tiles-fuji.fits')
            fuji_tiles=Table.read(fuji_tiles_filename)
            if release.upper()=='FUJI':
                surveys = ['sv1','sv3']
            elif release.upper()=='FUGU':
                surveys = ['main','sv1','sv3']
                guadalupe_tiles_filename=os.path.join(os.environ['DESI_SPECTRO_REDUX'],f'guadalupe/tiles-guadalupe.fits')
                guadalupe_tiles=Table.read(guadalupe_tiles_filename)
                redux_tiles=vstack([guadalupe_tiles,fuji_tiles])
            # All tiles in surveys (Main and/or SV)
            tile_arr = [load_tiles(tilesfile = tiles_filename.format(survey)) for survey in surveys]                
            tiles = vstack(tile_arr)
        
        # Only get DARK tiles in spectro redux
        dark_tiles = redux_tiles['PROGRAM']=='dark'
        tiles_in_surveys = np.isin(np.char.decode(redux_tiles['SURVEY']),surveys)
        redux_tiles = redux_tiles = redux_tiles[dark_tiles&tiles_in_surveys]
        
        tiles_in_redux = np.isin(tiles['TILEID'],redux_tiles['TILEID'])
        tiles = tiles[tiles_in_redux]
    return tiles
    
        
def get_catalog_dndzdm(cat,zbins=np.linspace(0,10,100+1), rmagbins=np.linspace(15,25,100+1),mask=None):
    """Get the redshift and magnitude distributions of a catalog.

    Args:
        cat (astropy.table.Table): data catalog.
        zbins (array): redshift bins.
        rmagbins (array): r-band magnitude bins.
        mask (array): mask to apply to the catalog before computing the distributions.

    Returns:
        dn_dzdm (array): redshift and magnitude distribution.
        zbins (array): redshift bins.
        rmagbins (array): r-band magnitude bins.
    """
    z=cat['Z']
    if 'RMAG' in cat.colnames:
        rmag = cat['RMAG']
    elif 'FLUX_R' in cat.colnames:
        rmag = 22.5-2.5*np.log10(cat['FLUX_R'])
    else:
        raise ValueError("Magnitude information could not be retrieved from data catalog")
    if mask is not None:
        z=z[mask]
        rmag = rmag[mask]
    dn_dzdm=np.histogram2d(z,rmag,bins=(zbins,rmagbins))[0]
    return dn_dzdm, zbins, rmagbins

def generate_random_fluxes(dist, z, zcenters, rmagcenters,invert=False):
    cdf=np.cumsum(dist,axis=1)
    cdf = np.divide(cdf,cdf[:,-1][:,None],where=cdf[:,-1][:,None]>0,out=np.zeros_like(cdf))
    dz = zcenters[1]-zcenters[0]
    mags=np.zeros(len(z))

    for i,z_bin in enumerate(zcenters):
        w_z = (z > z_bin-0.5*dz) & (z <= z_bin+0.5*dz)
        if np.sum(w_z)==0: continue
        rand = np.random.uniform(size=np.sum(w_z))
        if invert:
            rand = 1-rand
        mags[w_z]=np.interp(rand,cdf[i],rmagcenters)  
    if np.sum(mags==0)!=0:
        raise ValueError(f"Randomly generated magnitudes contain zeros.")
    return 10**((22.5-mags)/2.5)

def npass_pixmap_from_tiles(tiles,nside=1024):
    """Create a pixmap with the number of passes in each pixel.

    Args:
        tiles (astropy.table.Table): Table with the tiles.
        nside (int): HEALPix nside parameter.

    Returns:
        pixmap (astropy.table.Table): Table with the number of passes in each HEALPix pixel.
    """
    pixmap = Table()
    npix = hp.nside2npix(nside)
    pixmap["HPXPIXEL"] = np.arange(npix, dtype=int)
    thetas, phis = hp.pix2ang(nside, np.arange(npix), nest=True)
    pixmap["RA"], pixmap["DEC"] = np.degrees(phis), 90.0 - np.degrees(thetas)

    passes = np.unique(tiles["PASS"])
    npass = len(passes)
    # listing the area covered by 1, 2, ..., npass passes
    pixmap["TILEIDS"] = np.zeros((npix, npass), dtype=int)
    for i in range(len(tiles)):
        ipixs = tiles2pix(nside, tiles=Table(tiles[i]))
        pixmap["TILEIDS"][ipixs, tiles["PASS"][i]] = tiles["TILEID"][i]
    pixmap["NPASS"] = (pixmap["TILEIDS"] != 0).sum(axis=1)
    return pixmap
